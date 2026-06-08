# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
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
import src.models.block_mc as block_mc
from src.transforms.functional import rgb2ycbcr
from src.utils.stream_helper import get_padding_size, get_state_dict


def str2bool(value):
    return str(value).lower() in ("yes", "y", "true", "t", "1")


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
    def __init__(self, root, clip_len=5, crop_size=256):
        self.sequences = collect_png_sequences(root)
        self.clip_len = clip_len
        self.crop_size = crop_size
        self.index = []
        for seq_idx, frames in enumerate(self.sequences):
            for start in range(0, len(frames) - clip_len + 1):
                self.index.append((seq_idx, start))
        if len(self.index) == 0:
            raise ValueError("No clip can be formed. Reduce --clip_len or add more frames.")

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _read_rgb(path):
        image = Image.open(path).convert("RGB")
        array = np.asarray(image).astype("float32") / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def __getitem__(self, index):
        seq_idx, start = self.index[index]
        paths = self.sequences[seq_idx][start:start + self.clip_len]
        frames = [self._read_rgb(path) for path in paths]
        clip = torch.stack(frames, dim=0)
        _, _, height, width = clip.shape

        if self.crop_size > 0:
            crop_h = min(self.crop_size, height)
            crop_w = min(self.crop_size, width)
            if height > crop_h:
                top = random.randint(0, height - crop_h)
            else:
                top = 0
            if width > crop_w:
                left = random.randint(0, width - crop_w)
            else:
                left = 0
            clip = clip[:, :, top:top + crop_h, left:left + crop_w]

        _, _, height, width = clip.shape
        padding_l, padding_r, padding_t, padding_b = get_padding_size(height, width, 16)
        clip = F.pad(clip, (padding_l, padding_r, padding_t, padding_b), mode="replicate")
        yuv = rgb2ycbcr(clip)
        return yuv


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


def load_pretrained_hr(model, checkpoint_path):
    state_dict = get_state_dict(checkpoint_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    non_hr_missing = [key for key in missing if not key.startswith("ref_hierarchy.")]
    if unexpected:
        print(f"Unexpected keys in P-frame checkpoint: {unexpected}")
    if non_hr_missing:
        print(f"Missing non-HRS keys: {non_hr_missing}")
    print(f"Loaded P-frame checkpoint with {len(missing)} missing HRS/new keys.")


def is_stage1b_parameter(name):
    return name.startswith((
        "ref_hierarchy.",
        "feature_adaptor.",
        "feature_adaptor_I.",
        "align.",
        "context_fusion_net.",
        "temporal_prior_encoder.",
    ))


def is_stage3_lite_parameter(name):
    return is_stage1b_parameter(name) or name.startswith((
        "recon_generation_net.recon_conv.",
    ))


def summarize_trainable_parameters(model):
    summary = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group = name.split(".", 1)[0]
        summary[group] = summary.get(group, 0) + param.numel()
    return summary


def freeze_for_stage1(model, train_scope="safe"):
    for param in model.parameters():
        param.requires_grad = False

    if train_scope == "stage1b":
        for name, param in model.named_parameters():
            param.requires_grad = is_stage1b_parameter(name)
        return [param for param in model.parameters() if param.requires_grad]

    if train_scope == "stage3_lite":
        for name, param in model.named_parameters():
            param.requires_grad = is_stage3_lite_parameter(name)
        return [param for param in model.parameters() if param.requires_grad]

    for name, param in model.ref_hierarchy.named_parameters():
        if train_scope == "all":
            param.requires_grad = True
        elif train_scope == "safe":
            param.requires_grad = name.startswith(("hist_fusion", "to_me_delta"))
        elif train_scope == "delta":
            param.requires_grad = name.startswith("to_me_delta")
        elif train_scope == "fusion":
            param.requires_grad = name.startswith("hist_fusion")
        else:
            raise ValueError(f"Unsupported train_scope: {train_scope}")
    return [param for param in model.parameters() if param.requires_grad]


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


def save_checkpoint(path, model, optimizer, step, epoch, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "args": vars(args),
    }, path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal Stage 1 trainer for DMC_HR HRS modules."
    )
    parser.add_argument("--train_root", type=str, required=True,
                        help="Root containing PNG frame sequences.")
    parser.add_argument("--model_path_i", type=str, required=True)
    parser.add_argument("--model_path_p", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume_optimizer", type=str2bool, default=True,
                        help="Load optimizer state from --resume. Disable this when train_scope changes.")
    parser.add_argument("--required_resume_step", type=int, default=None,
                        help="Fail if the resume checkpoint step does not match this value.")
    parser.add_argument("--save_dir", type=str, default="checkpoints_stage1_hr")
    parser.add_argument("--clip_len", type=int, default=5)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--worker", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--max_new_steps", type=int, default=None,
                        help="If set, train this many optimizer steps from the current/resumed step.")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--lambda_bpp", type=float, default=0.0)
    parser.add_argument("--lambda_mc", type=float, default=0.0)
    parser.add_argument("--lambda_identity", type=float, default=0.1)
    parser.add_argument("--lambda_distill", type=float, default=5.0)
    parser.add_argument("--beta_balance", type=float, default=0.0)
    parser.add_argument("--train_scope", type=str, default="delta",
                        choices=["safe", "delta", "fusion", "all", "stage1b", "stage3_lite"])
    parser.add_argument("--me_delta_scale", type=float, default=0.02,
                        help="Maximum HRS residual strength added to the ME reference.")
    parser.add_argument("--q_index_i", type=int, default=32)
    parser.add_argument("--q_index_p", type=int, default=32)
    parser.add_argument("--rate_gop_size", type=int, default=8, choices=[4, 8])
    parser.add_argument("--cuda", type=str2bool, default=True)
    parser.add_argument("--force_torch_warp", type=str2bool, default=True,
                        help="Use differentiable PyTorch grid_sample warp during training.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if args.force_torch_warp:
        block_mc.CUSTOMIZED_CUDA = False
        print("Training uses differentiable PyTorch warp instead of custom block_mc CUDA.", flush=True)
    dataset = PNGSequenceDataset(args.train_root, args.clip_len, args.crop_size)
    print(
        f"Found {len(dataset.sequences)} sequences and {len(dataset)} clips "
        f"under {args.train_root}.",
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
    for param in i_frame_net.parameters():
        param.requires_grad = False

    p_frame_net = build_dmc_hr(inplace=False, me_delta_scale=args.me_delta_scale).to(device)
    load_pretrained_hr(p_frame_net, args.model_path_p)
    trainable_params = freeze_for_stage1(p_frame_net, args.train_scope)
    trainable_count = sum(param.numel() for param in trainable_params)
    print(f"Trainable parameters ({args.train_scope}): {trainable_count:,}", flush=True)
    for group, count in sorted(summarize_trainable_parameters(p_frame_net).items()):
        print(f"  trainable[{group}]: {count:,}", flush=True)
    print(
        "Stage 1 v4 regularization: "
        f"me_delta_scale={args.me_delta_scale}, "
        f"lambda_bpp={args.lambda_bpp}, lambda_mc={args.lambda_mc}, "
        f"lambda_identity={args.lambda_identity}, lambda_distill={args.lambda_distill}, "
        f"beta_balance={args.beta_balance}",
        flush=True,
    )
    p_frame_net.train()
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    teacher_net = DMC(inplace=False).to(device)
    teacher_net.load_state_dict(get_state_dict(args.model_path_p))
    teacher_net.eval()
    for param in teacher_net.parameters():
        param.requires_grad = False

    start_epoch = 0
    step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        missing, unexpected = p_frame_net.load_state_dict(checkpoint["model"], strict=False)
        if unexpected:
            print(f"Unexpected keys in resume checkpoint: {unexpected}", flush=True)
        non_hr_missing = [key for key in missing if not key.startswith("ref_hierarchy.")]
        hr_missing = [key for key in missing if key.startswith("ref_hierarchy.")]
        if non_hr_missing:
            print(f"Missing non-HRS keys in resume checkpoint: {non_hr_missing}", flush=True)
        if hr_missing:
            print(f"Initialized {len(hr_missing)} new HRS keys from defaults.", flush=True)
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
                f"Resume checkpoint step is {step}, but --required_resume_step "
                f"is {args.required_resume_step}."
            )
        start_epoch = int(checkpoint.get("epoch", 0))
        if args.max_new_steps is not None:
            start_epoch = 0
        print(f"Resumed model from {args.resume} at step {step}.", flush=True)

    if args.max_new_steps is not None:
        target_step = step + args.max_new_steps
    else:
        target_step = args.max_steps
    if target_step <= step:
        raise ValueError(
            f"Training target_step={target_step} must be greater than current step={step}. "
            "Use --max_new_steps when resuming for additional training."
        )
    print(
        f"Training target step: {target_step} "
        f"(current step {step}, max_new_steps={args.max_new_steps}).",
        flush=True,
    )

    index_map = [0, 1, 0, 2, 0, 2, 0, 2]
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        print(f"Starting epoch {epoch + 1}/{args.epochs} at step {step}.", flush=True)
        for clip in loader:
            if step == 0:
                print("First training batch loaded; building initial DPB.", flush=True)
            clip = clip.to(device, non_blocking=True)
            frames = clip.unbind(dim=1)
            dpb = build_initial_dpb(i_frame_net, frames[0], args.q_index_i)
            teacher_dpb = {key: value.clone() if torch.is_tensor(value) else value
                           for key, value in dpb.items()}
            if step == 0:
                print("Initial I-frame DPB built; running first P-frame update.", flush=True)
            optimizer.zero_grad(set_to_none=True)

            total_loss = 0.0
            total_mse = 0.0
            total_bpp = 0.0
            total_balance = 0.0
            total_mc = 0.0
            total_identity = 0.0
            total_distill = 0.0
            p_count = 0

            for frame_idx, frame in enumerate(frames[1:], start=1):
                fa_idx = index_map[frame_idx % args.rate_gop_size]
                if step == 0:
                    print(f"Forward P-frame {frame_idx}/{len(frames) - 1}.", flush=True)
                with torch.no_grad():
                    teacher_out = teacher_net.forward_one_frame(
                        frame, teacher_dpb, q_index=args.q_index_p, fa_idx=fa_idx
                    )
                out = p_frame_net.forward_one_frame(
                    frame, dpb, q_index=args.q_index_p, fa_idx=fa_idx
                )
                recon = out["dpb"]["ref_frame"]
                teacher_recon = teacher_out["dpb"]["ref_frame"]
                mse = F.mse_loss(recon, frame)
                aux = out.get("aux", {})
                mc_loss = F.l1_loss(aux["warpframe"], frame) if "warpframe" in aux else 0.0
                identity_loss = F.l1_loss(aux["me_ref"], dpb["ref_frame"]) if "me_ref" in aux else 0.0
                distill_loss = F.mse_loss(recon, teacher_recon)
                pixel_num = frame.shape[-2] * frame.shape[-1]
                bpp = out["bit"] / (pixel_num * frame.shape[0])
                balance = out.get("balance_loss")
                if balance is None:
                    balance = torch.zeros((), dtype=frame.dtype, device=device)
                loss = (
                    mse
                    + args.lambda_bpp * bpp
                    + args.lambda_mc * mc_loss
                    + args.lambda_identity * identity_loss
                    + args.lambda_distill * distill_loss
                    + args.beta_balance * balance
                )

                total_loss = total_loss + loss
                total_mse = total_mse + mse.detach()
                total_bpp = total_bpp + bpp.detach()
                total_balance = total_balance + balance.detach()
                total_mc = total_mc + (mc_loss.detach() if torch.is_tensor(mc_loss) else mc_loss)
                total_identity = total_identity + (
                    identity_loss.detach() if torch.is_tensor(identity_loss) else identity_loss
                )
                total_distill = total_distill + distill_loss.detach()
                p_count += 1
                dpb = {key: value.detach() if torch.is_tensor(value) else value
                       for key, value in out["dpb"].items()}
                teacher_dpb = {
                    key: value.detach() if torch.is_tensor(value) else value
                    for key, value in teacher_out["dpb"].items()
                }

            total_loss = total_loss / max(p_count, 1)
            if step == 0:
                print("Forward pass finished; starting backward.", flush=True)
            total_loss.backward()
            if step == 0:
                print("Backward finished; applying optimizer step.", flush=True)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            if step == 0:
                print("Optimizer step finished.", flush=True)
            step += 1

            if step % args.log_interval == 0 or step == 1:
                print(
                    f"step {step} loss {total_loss.item():.6f} "
                    f"mse {(total_mse / p_count).item():.6f} "
                    f"bpp {(total_bpp / p_count).item():.6f} "
                    f"mc {(total_mc / p_count).item():.6f} "
                    f"id {(total_identity / p_count).item():.6f} "
                    f"distill {(total_distill / p_count).item():.6f} "
                    f"balance {(total_balance / p_count).item():.6f}"
                )

            if step % args.save_interval == 0:
                save_checkpoint(save_dir / f"stage1_hr_step{step}.pth.tar",
                                p_frame_net, optimizer, step, epoch, args)

            if step >= target_step:
                break
        if step >= target_step:
            break

    save_checkpoint(save_dir / "stage1_hr_latest.pth.tar",
                    p_frame_net, optimizer, step, args.epochs, args)


if __name__ == "__main__":
    main()
