# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def find_project_root(start_path):
    for parent in [start_path, *start_path.parents]:
        if (parent / "src").is_dir():
            return parent
    raise RuntimeError(f"Cannot find project root containing src/ from {start_path}.")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.image_model import DMCI
from src.models.video_model import DMC, DMC_HR
from src.nvcx.wrappers import NeuralWrapper
from src.transforms.functional import rgb2ycbcr
from src.utils.stream_helper import get_padding_size, get_state_dict


def str2bool(value):
    return str(value).lower() in ("yes", "y", "true", "t", "1")


def parse_int_list(text):
    if isinstance(text, (list, tuple)):
        return [int(x) for x in text]
    return [int(x.strip()) for x in str(text).replace(",", " ").split() if x.strip()]


def sample_q_index(q_indexes, mode):
    if mode == "random":
        return random.choice(q_indexes)
    if mode == "cycle":
        return q_indexes[sample_q_index.counter % len(q_indexes)]
    raise ValueError(f"Unsupported q_sample_mode: {mode}")


sample_q_index.counter = 0


def collect_png_sequences(root):
    root = Path(root)
    sequences = []
    for folder in sorted([p for p in root.rglob("*") if p.is_dir()]):
        frames = sorted(folder.glob("*.png"))
        if len(frames) >= 2:
            sequences.append(frames)
    if len(sequences) == 0:
        frames = sorted(root.glob("*.png"))
        if len(frames) >= 2:
            sequences.append(frames)
    if len(sequences) == 0:
        raise ValueError(f"No PNG sequences with at least two frames found under {root}.")
    return sequences


class PNGSequenceDataset(Dataset):
    def __init__(self, root, clip_len=2, crop_size=128, temporal_stride_choices=(1,)):
        self.sequences = collect_png_sequences(root)
        self.clip_len = clip_len
        self.crop_size = crop_size
        self.temporal_stride_choices = tuple(temporal_stride_choices)
        self.index = []
        max_stride = max(self.temporal_stride_choices)
        span = 1 + (clip_len - 1) * max_stride
        for seq_idx, frames in enumerate(self.sequences):
            for start in range(0, len(frames) - span + 1):
                self.index.append((seq_idx, start))
        if len(self.index) == 0:
            raise ValueError("No clip can be formed. Reduce --clip_len/--temporal_strides.")

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _read_rgb(path):
        image = Image.open(path).convert("RGB")
        array = np.asarray(image).astype("float32") / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def __getitem__(self, index):
        seq_idx, start = self.index[index]
        frames_all = self.sequences[seq_idx]
        stride = random.choice(self.temporal_stride_choices)
        frame_ids = [start + i * stride for i in range(self.clip_len)]
        paths = [frames_all[i] for i in frame_ids]
        frames = [self._read_rgb(path) for path in paths]
        clip = torch.stack(frames, dim=0)
        _, _, height, width = clip.shape

        if self.crop_size > 0:
            crop_h = min(self.crop_size, height)
            crop_w = min(self.crop_size, width)
            top = random.randint(0, height - crop_h) if height > crop_h else 0
            left = random.randint(0, width - crop_w) if width > crop_w else 0
            clip = clip[:, :, top:top + crop_h, left:left + crop_w]

        _, _, height, width = clip.shape
        padding_l, padding_r, padding_t, padding_b = get_padding_size(height, width, 16)
        clip = F.pad(clip, (padding_l, padding_r, padding_t, padding_b), mode="replicate")
        return rgb2ycbcr(clip)


def set_me_delta_scale(model, scale):
    if hasattr(model, "me_delta_scale"):
        model.me_delta_scale = scale
    if hasattr(model, "ref_hierarchy") and hasattr(model.ref_hierarchy, "me_delta_scale"):
        model.ref_hierarchy.me_delta_scale = scale


def build_dmc_hr(inplace=False, me_delta_scale=0.02):
    try:
        model = DMC_HR(inplace=inplace, me_delta_scale=me_delta_scale)
    except TypeError as exc:
        if "me_delta_scale" not in str(exc):
            raise
        model = DMC_HR(inplace=inplace)
        set_me_delta_scale(model, me_delta_scale)
    return model


def build_initial_dpb(i_frame_net, first_frame, q_index_i):
    with torch.no_grad():
        result = i_frame_net.encode(first_frame, q_index_i)
    return {
        "ref_frame": result["x_hat"].detach(),
        "ref_feature": None,
        "ref_mv_feature": None,
        "ref_y": None,
        "ref_mv_y": None,
    }


def load_core_checkpoint(core, checkpoint_path):
    state_dict = get_state_dict(checkpoint_path)
    missing, unexpected = core.load_state_dict(state_dict, strict=False)
    non_hr_missing = [key for key in missing if not key.startswith("ref_hierarchy.")]
    if unexpected:
        print(f"Unexpected keys in core checkpoint: {unexpected}", flush=True)
    if non_hr_missing:
        print(f"Missing non-HRS core keys: {non_hr_missing}", flush=True)
    print(f"Loaded core checkpoint with {len(missing)} missing HRS/new keys.", flush=True)


def set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def save_checkpoint(path, wrapper, optimizer, step, epoch, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": wrapper.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "args": vars(args),
    }, path)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 post-only trainer for NeuralWrapper.")
    parser.add_argument("--train_root", type=str, required=True)
    parser.add_argument("--model_path_i", type=str, required=True)
    parser.add_argument("--core_checkpoint", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume_optimizer", type=str2bool, default=True)
    parser.add_argument("--required_resume_step", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="checkpoints_stage2_post_only")
    parser.add_argument("--checkpoint_prefix", type=str, default="stage2_post")
    parser.add_argument("--train_scope", type=str, default="post_only",
                        choices=["post_only", "pre_post"])
    parser.add_argument("--post_as_ref", type=str2bool, default=False)
    parser.add_argument("--clip_len", type=int, default=2)
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--temporal_strides", type=str, default="1")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--max_new_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lambda_bpp", type=float, default=0.0)
    parser.add_argument("--lambda_identity", type=float, default=0.05)
    parser.add_argument("--lambda_residual", type=float, default=0.01)
    parser.add_argument("--lambda_core_distill", type=float, default=0.1)
    parser.add_argument("--lambda_pre_delta", type=float, default=0.0)
    parser.add_argument("--me_delta_scale", type=float, default=0.02)
    parser.add_argument("--q_indexes", type=str, default="")
    parser.add_argument("--q_sample_mode", type=str, default="random", choices=["random", "cycle"])
    parser.add_argument("--q_index_i_same_as_p", type=str2bool, default=False)
    parser.add_argument("--q_index_i", type=int, default=32)
    parser.add_argument("--q_index_p", type=int, default=32)
    parser.add_argument("--rate_gop_size", type=int, default=8, choices=[4, 8])
    parser.add_argument("--cuda", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    q_indexes = parse_int_list(args.q_indexes) if args.q_indexes else [args.q_index_p]
    temporal_strides = parse_int_list(args.temporal_strides)
    if len(q_indexes) == 0:
        raise ValueError("--q_indexes must contain at least one q index.")
    if len(temporal_strides) == 0:
        raise ValueError("--temporal_strides must contain at least one stride.")

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    dataset = PNGSequenceDataset(
        args.train_root,
        args.clip_len,
        args.crop_size,
        temporal_stride_choices=temporal_strides,
    )
    print(
        f"Found {len(dataset.sequences)} sequences and {len(dataset)} clips "
        f"under {args.train_root}. clip_len={args.clip_len}, temporal_strides={temporal_strides}",
        flush=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.worker,
        pin_memory=device == "cuda",
        drop_last=True,
    )

    i_frame_net = DMCI(inplace=False).to(device)
    i_frame_net.load_state_dict(get_state_dict(args.model_path_i))
    i_frame_net.eval()
    set_requires_grad(i_frame_net, False)

    core = build_dmc_hr(inplace=False, me_delta_scale=args.me_delta_scale).to(device)
    load_core_checkpoint(core, args.core_checkpoint)
    core.eval()
    set_requires_grad(core, False)

    wrapper = NeuralWrapper(
        core,
        qp_num=DMC.get_qp_num(),
        inplace=False,
        use_pre=args.train_scope == "pre_post",
        use_post=True,
        post_as_ref=args.post_as_ref,
    ).to(device)
    set_requires_grad(wrapper.core, False)
    set_requires_grad(wrapper.pre, args.train_scope == "pre_post")
    set_requires_grad(wrapper.post, True)
    wrapper.train()
    wrapper.core.eval()
    if args.train_scope == "pre_post":
        wrapper.pre.train()
    else:
        wrapper.pre.eval()
    wrapper.post.train()

    trainable_params = [param for param in wrapper.parameters() if param.requires_grad]
    print(f"Trainable wrapper parameters ({args.train_scope}): "
          f"{sum(param.numel() for param in trainable_params):,}", flush=True)
    print(
        "Stage 2 wrapper regularization: "
        f"lambda_bpp={args.lambda_bpp}, "
        f"lambda_identity={args.lambda_identity}, "
        f"lambda_residual={args.lambda_residual}, "
        f"lambda_core_distill={args.lambda_core_distill}, "
        f"lambda_pre_delta={args.lambda_pre_delta}, "
        f"post_as_ref={args.post_as_ref}, q_indexes={q_indexes}",
        flush=True,
    )
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    start_epoch = 0
    step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        wrapper.load_state_dict(checkpoint["model"], strict=True)
        if args.resume_optimizer:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
                print("Loaded optimizer state from resume checkpoint.", flush=True)
            except (KeyError, ValueError) as exc:
                print(f"Could not load optimizer state; using a fresh optimizer. Reason: {exc}",
                      flush=True)
        else:
            print("Resume optimizer is disabled; using a fresh optimizer.", flush=True)
        step = int(checkpoint.get("step", 0))
        if args.required_resume_step is not None and step != args.required_resume_step:
            raise ValueError(
                f"Resume checkpoint step is {step}, but --required_resume_step is "
                f"{args.required_resume_step}."
            )
        start_epoch = int(checkpoint.get("epoch", 0))
        if args.max_new_steps is not None:
            start_epoch = 0
        print(f"Resumed Stage 2 wrapper from {args.resume} at step {step}.", flush=True)

    if args.max_new_steps is not None:
        target_step = step + args.max_new_steps
    else:
        target_step = args.max_steps
    if target_step <= step:
        raise ValueError("Target step must be greater than current step.")
    print(
        f"Training target step: {target_step} "
        f"(current step {step}, max_new_steps={args.max_new_steps}).",
        flush=True,
    )

    index_map = [0, 1, 0, 2, 0, 2, 0, 2]
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        print(f"Starting Stage 2 epoch {epoch + 1}/{args.epochs} at step {step}.", flush=True)
        for clip in loader:
            q_index_p = sample_q_index(q_indexes, args.q_sample_mode)
            sample_q_index.counter += 1
            q_index_i = q_index_p if args.q_index_i_same_as_p else args.q_index_i
            clip = clip.to(device, non_blocking=True)
            frames = clip.unbind(dim=1)
            dpb = build_initial_dpb(i_frame_net, frames[0], q_index_i)
            optimizer.zero_grad(set_to_none=True)

            total_loss = 0.0
            total_mse = 0.0
            total_identity = 0.0
            total_residual = 0.0
            total_core_distill = 0.0
            total_pre_delta = 0.0
            total_bpp = 0.0
            p_count = 0

            for frame_idx, frame in enumerate(frames[1:], start=1):
                fa_idx = index_map[frame_idx % args.rate_gop_size]
                out = wrapper.forward_one_frame(frame, dpb, q_index_p, fa_idx)
                x_tilde = out["x_tilde"]
                x_hat_core = out["x_hat_core"].detach()
                mse = F.mse_loss(x_tilde, frame)
                identity_loss = F.l1_loss(x_tilde, x_hat_core)
                residual_loss = F.smooth_l1_loss(x_tilde, x_hat_core)
                core_distill = F.mse_loss(x_tilde, x_hat_core)
                pixel_num = frame.shape[-2] * frame.shape[-1]
                bpp = out["bit"] / (pixel_num * frame.shape[0])
                pre_delta = torch.zeros((), dtype=frame.dtype, device=device)
                delta_pre = out.get("pre_aux", {}).get("delta_pre")
                if delta_pre is not None:
                    pre_delta = F.smooth_l1_loss(delta_pre, torch.zeros_like(delta_pre))
                loss = (
                    mse
                    + args.lambda_bpp * bpp
                    + args.lambda_identity * identity_loss
                    + args.lambda_residual * residual_loss
                    + args.lambda_core_distill * core_distill
                    + args.lambda_pre_delta * pre_delta
                )

                total_loss = total_loss + loss
                total_mse = total_mse + mse.detach()
                total_identity = total_identity + identity_loss.detach()
                total_residual = total_residual + residual_loss.detach()
                total_core_distill = total_core_distill + core_distill.detach()
                total_pre_delta = total_pre_delta + pre_delta.detach()
                total_bpp = total_bpp + bpp.detach()
                p_count += 1
                dpb = {
                    key: value.detach() if torch.is_tensor(value) else value
                    for key, value in out["dpb"].items()
                }

            total_loss = total_loss / max(p_count, 1)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            step += 1

            if step % args.log_interval == 0 or step == 1:
                print(
                    f"step {step} q {q_index_p} loss {total_loss.item():.6f} "
                    f"mse {(total_mse / p_count).item():.6f} "
                    f"bpp {(total_bpp / p_count).item():.6f} "
                    f"id {(total_identity / p_count).item():.6f} "
                    f"res {(total_residual / p_count).item():.6f} "
                    f"core {(total_core_distill / p_count).item():.6f} "
                    f"pre_delta {(total_pre_delta / p_count).item():.6f}",
                    flush=True,
                )

            if step % args.save_interval == 0:
                save_checkpoint(save_dir / f"{args.checkpoint_prefix}_step{step}.pth.tar",
                                wrapper, optimizer, step, epoch, args)

            if step >= target_step:
                break
        if step >= target_step:
            break

    save_checkpoint(save_dir / f"{args.checkpoint_prefix}_latest.pth.tar",
                    wrapper, optimizer, step, args.epochs, args)


if __name__ == "__main__":
    main()
