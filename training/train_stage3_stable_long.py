# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import dataclasses
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
    def __init__(self, root, clip_len=6, crop_size=128, temporal_stride_choices=(1,)):
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
        frames = [self._read_rgb(frames_all[i]) for i in frame_ids]
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


def detach_dpb(dpb):
    if dataclasses.is_dataclass(dpb):
        return type(dpb)(**{
            field.name: detach_dpb(getattr(dpb, field.name))
            for field in dataclasses.fields(dpb)
        })
    if isinstance(dpb, dict):
        return {key: detach_dpb(value) for key, value in dpb.items()}
    if isinstance(dpb, list):
        return [detach_dpb(value) for value in dpb]
    if torch.is_tensor(dpb):
        return dpb.detach()
    return dpb


def clone_dpb(dpb):
    if dataclasses.is_dataclass(dpb):
        return type(dpb)(**{
            field.name: clone_dpb(getattr(dpb, field.name))
            for field in dataclasses.fields(dpb)
        })
    if isinstance(dpb, dict):
        return {key: clone_dpb(value) for key, value in dpb.items()}
    if isinstance(dpb, list):
        return [clone_dpb(value) for value in dpb]
    if torch.is_tensor(dpb):
        return dpb.clone()
    return dpb


def load_pretrained_hr(model, checkpoint_path):
    state_dict = get_state_dict(checkpoint_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    non_hr_missing = [key for key in missing if not key.startswith("ref_hierarchy.")]
    if unexpected:
        print(f"Unexpected keys in P-frame checkpoint: {unexpected}", flush=True)
    if non_hr_missing:
        print(f"Missing non-HRS keys: {non_hr_missing}", flush=True)
    print(f"Loaded P-frame checkpoint with {len(missing)} missing HRS/new keys.", flush=True)


def set_me_delta_scale(model, scale):
    if hasattr(model, "me_delta_scale"):
        model.me_delta_scale = scale
    if hasattr(model, "ref_hierarchy") and hasattr(model.ref_hierarchy, "me_delta_scale"):
        model.ref_hierarchy.me_delta_scale = scale


def build_dmc_hr(
    inplace=False,
    me_delta_scale=0.02,
    max_alpha_hist=0.05,
    max_alpha_expert=0.03,
    hrs_gate_init=6.0,
):
    try:
        model = DMC_HR(
            inplace=inplace,
            me_delta_scale=me_delta_scale,
            max_alpha_hist=max_alpha_hist,
            max_alpha_expert=max_alpha_expert,
            hrs_gate_init=hrs_gate_init,
        )
    except TypeError as exc:
        if not any(key in str(exc) for key in (
            "me_delta_scale",
            "max_alpha_hist",
            "max_alpha_expert",
            "hrs_gate_init",
        )):
            raise
        model = DMC_HR(inplace=inplace)
        set_me_delta_scale(model, me_delta_scale)
    return model


def is_hierarchy_gate_parameter(name):
    return name.startswith((
        "ref_hierarchy.global_strength",
        "ref_hierarchy.q_strength.",
    ))


def is_hrs_delta_only_parameter(name):
    return is_hierarchy_gate_parameter(name) or name.startswith("ref_hierarchy.to_me_delta.")


def is_hrs_hist_delta_parameter(name):
    return is_hrs_delta_only_parameter(name) or name.startswith((
        "ref_hierarchy.hist_fusion1.",
        "ref_hierarchy.hist_fusion2.",
        "ref_hierarchy.hist_fusion3.",
    ))


def is_hrs_router_delta_parameter(name):
    return is_hrs_hist_delta_parameter(name) or name.startswith((
        "ref_hierarchy.proj_main.",
        "ref_hierarchy.proj_gate.",
        "ref_hierarchy.q_embed.",
        "ref_hierarchy.age_embed.",
        "ref_hierarchy.fa_embed.",
        "ref_hierarchy.router.",
        "ref_hierarchy.expert_m.",
        "ref_hierarchy.expert_g.",
        "ref_hierarchy.expert_o.",
    ))


def is_stage1a_hrs_parameter(name):
    return name.startswith("ref_hierarchy.")


def is_stage1b_light_parameter(name):
    return name.startswith((
        "ref_hierarchy.",
        "feature_adaptor.",
        "feature_adaptor_I.",
        "align.",
        "context_fusion_net.",
        "temporal_prior_encoder.",
    ))


def is_stage3_joint_light_parameter(name):
    return name.startswith((
        "ref_hierarchy.",
        "feature_adaptor.",
        "feature_adaptor_I.",
        "context_fusion_net.conv1_out.",
        "context_fusion_net.res_block1_out.",
        "recon_generation_net.unet_2.",
        "recon_generation_net.recon_conv.",
    ))


def is_stable_core_parameter(name):
    return name.startswith((
        "ref_hierarchy.",
        "feature_adaptor.",
        "feature_adaptor_I.",
        "align.",
        "context_fusion_net.",
    ))


def is_prior_light_parameter(name):
    return is_stable_core_parameter(name) or name.startswith((
        "temporal_prior_encoder.",
        "y_prior_fusion_adaptor_0.",
        "y_prior_fusion_adaptor_1.",
        "y_prior_fusion.",
    ))


def is_recon_tail_parameter(name):
    return is_prior_light_parameter(name) or name.startswith((
        "recon_generation_net.recon_conv.",
    ))


def select_trainable_parameters(model, train_scope):
    for param in model.parameters():
        param.requires_grad = False

    if train_scope == "hrs_delta_only":
        predicate = is_hrs_delta_only_parameter
    elif train_scope == "hrs_hist_delta":
        predicate = is_hrs_hist_delta_parameter
    elif train_scope == "hrs_router_delta":
        predicate = is_hrs_router_delta_parameter
    elif train_scope == "stage1a_hrs":
        predicate = is_stage1a_hrs_parameter
    elif train_scope == "stage1b_light":
        predicate = is_stage1b_light_parameter
    elif train_scope == "stage3_joint_light":
        predicate = is_stage3_joint_light_parameter
    elif train_scope == "stable_core":
        predicate = is_stable_core_parameter
    elif train_scope == "prior_light":
        predicate = is_prior_light_parameter
    elif train_scope == "recon_tail":
        predicate = is_recon_tail_parameter
    else:
        raise ValueError(f"Unsupported train_scope: {train_scope}")

    for name, param in model.named_parameters():
        param.requires_grad = predicate(name)
    return [param for param in model.parameters() if param.requires_grad]


def summarize_trainable_parameters(model):
    summary = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group = name.split(".", 1)[0]
        summary[group] = summary.get(group, 0) + param.numel()
    return summary


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


def get_scheduled_weight(start, end, local_step, total_steps):
    if total_steps <= 1:
        return end
    progress = min(max(local_step / float(total_steps - 1), 0.0), 1.0)
    return start + (end - start) * progress


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stable long-clip Stage 3 trainer for DMC_HR."
    )
    parser.add_argument("--train_root", type=str, required=True)
    parser.add_argument("--model_path_i", type=str, required=True)
    parser.add_argument("--model_path_p", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume_optimizer", type=str2bool, default=False)
    parser.add_argument("--required_resume_step", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="checkpoints_stage3_stable_long")
    parser.add_argument("--clip_len", type=int, default=6)
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--temporal_strides", type=str, default="1")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--worker", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--max_new_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=5e-8)
    parser.add_argument("--train_scope", type=str, default="stable_core",
                        choices=[
                            "hrs_delta_only",
                            "hrs_hist_delta",
                            "hrs_router_delta",
                            "stage1a_hrs",
                            "stage1b_light",
                            "stage3_joint_light",
                            "stable_core",
                            "prior_light",
                            "recon_tail",
                        ])
    parser.add_argument("--checkpoint_prefix", type=str, default="stage3_stable")
    parser.add_argument("--q_indexes", type=str, default="0 21 42 63")
    parser.add_argument("--q_sample_mode", type=str, default="random", choices=["random", "cycle"])
    parser.add_argument("--q_index_i_same_as_p", type=str2bool, default=True)
    parser.add_argument("--q_index_i", type=int, default=32)
    parser.add_argument("--lambda_bpp", type=float, default=0.0)
    parser.add_argument("--lambda_mc", type=float, default=0.0)
    parser.add_argument("--lambda_identity", type=float, default=0.1)
    parser.add_argument("--lambda_distill_start", type=float, default=2.0)
    parser.add_argument("--lambda_distill_end", type=float, default=0.25)
    parser.add_argument("--lambda_log_bpp_distill", type=float, default=0.0)
    parser.add_argument("--lambda_bit_ceiling", type=float, default=0.0)
    parser.add_argument("--bit_ceiling_ratio", type=float, default=1.02)
    parser.add_argument("--lambda_me_reg", type=float, default=0.0)
    parser.add_argument("--lambda_feature_reg", type=float, default=0.0)
    parser.add_argument("--lambda_expert_reg", type=float, default=0.0)
    parser.add_argument("--beta_balance", type=float, default=0.0)
    parser.add_argument("--late_frame_gamma", type=float, default=1.0,
                        help=">1 weights later P-frames more strongly inside each clip.")
    parser.add_argument("--detach_dpb", type=str2bool, default=True,
                        help="Keep closed-loop forward states but avoid full BPTT memory growth.")
    parser.add_argument("--me_delta_scale", type=float, default=0.02)
    parser.add_argument("--max_alpha_hist", type=float, default=0.05)
    parser.add_argument("--max_alpha_expert", type=float, default=0.03)
    parser.add_argument("--hrs_gate_init", type=float, default=6.0)
    parser.add_argument("--rate_gop_size", type=int, default=8, choices=[4, 8])
    parser.add_argument("--cuda", type=str2bool, default=True)
    parser.add_argument("--force_torch_warp", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    q_indexes = parse_int_list(args.q_indexes)
    temporal_strides = parse_int_list(args.temporal_strides)
    if len(q_indexes) == 0:
        raise ValueError("--q_indexes must contain at least one q index.")

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if args.force_torch_warp:
        block_mc.CUSTOMIZED_CUDA = False
        print("Training uses differentiable PyTorch warp instead of custom block_mc CUDA.",
              flush=True)

    dataset = PNGSequenceDataset(
        args.train_root,
        clip_len=args.clip_len,
        crop_size=args.crop_size,
        temporal_stride_choices=temporal_strides,
    )
    print(
        f"Found {len(dataset.sequences)} sequences and {len(dataset)} clips under "
        f"{args.train_root}. clip_len={args.clip_len}, temporal_strides={temporal_strides}",
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

    p_frame_net = build_dmc_hr(
        inplace=False,
        me_delta_scale=args.me_delta_scale,
        max_alpha_hist=args.max_alpha_hist,
        max_alpha_expert=args.max_alpha_expert,
        hrs_gate_init=args.hrs_gate_init,
    ).to(device)
    load_pretrained_hr(p_frame_net, args.model_path_p)

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
        step = int(checkpoint.get("step", 0))
        if args.required_resume_step is not None and step != args.required_resume_step:
            raise ValueError(
                f"Resume checkpoint step is {step}, but --required_resume_step is "
                f"{args.required_resume_step}."
            )
        start_epoch = int(checkpoint.get("epoch", 0))
        if args.max_new_steps is not None:
            start_epoch = 0
        print(f"Resumed model from {args.resume} at step {step}.", flush=True)

    trainable_params = select_trainable_parameters(p_frame_net, args.train_scope)
    trainable_count = sum(param.numel() for param in trainable_params)
    if trainable_count == 0:
        raise RuntimeError(f"No trainable parameters selected for scope {args.train_scope}.")
    print(f"Trainable parameters ({args.train_scope}): {trainable_count:,}", flush=True)
    for group, count in sorted(summarize_trainable_parameters(p_frame_net).items()):
        print(f"  trainable[{group}]: {count:,}", flush=True)

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    if args.resume is not None and args.resume_optimizer:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("Loaded optimizer state from resume checkpoint.", flush=True)
        except (KeyError, ValueError) as exc:
            print(f"Could not load optimizer state; using a fresh optimizer. Reason: {exc}",
                  flush=True)

    p_frame_net.train()

    teacher_net = DMC(inplace=False).to(device)
    teacher_net.load_state_dict(get_state_dict(args.model_path_p))
    teacher_net.eval()
    for param in teacher_net.parameters():
        param.requires_grad = False

    if args.max_new_steps is not None:
        target_step = step + args.max_new_steps
        schedule_steps = args.max_new_steps
    else:
        target_step = args.max_steps
        schedule_steps = max(target_step - step, 1)
    if target_step <= step:
        raise ValueError("Target step must be greater than current step.")
    print(
        f"Training target step: {target_step}; current step={step}; "
        f"q_indexes={q_indexes}; distill {args.lambda_distill_start}->{args.lambda_distill_end}",
        flush=True,
    )

    index_map = [0, 1, 0, 2, 0, 2, 0, 2]
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        print(f"Starting epoch {epoch + 1}/{args.epochs} at step {step}.", flush=True)
        for clip in loader:
            local_step = step - (target_step - schedule_steps)
            q_index_p = sample_q_index(q_indexes, args.q_sample_mode)
            sample_q_index.counter += 1
            q_index_i = q_index_p if args.q_index_i_same_as_p else args.q_index_i
            distill_weight = get_scheduled_weight(
                args.lambda_distill_start,
                args.lambda_distill_end,
                max(local_step, 0),
                schedule_steps,
            )

            clip = clip.to(device, non_blocking=True)
            frames = clip.unbind(dim=1)
            dpb = build_initial_dpb(i_frame_net, frames[0], q_index_i)
            teacher_dpb = clone_dpb(dpb)
            optimizer.zero_grad(set_to_none=True)

            weighted_losses = []
            stats = {
                "mse": 0.0,
                "bpp": 0.0,
                "teacher_bpp": 0.0,
                "log_bpp": 0.0,
                "bit_ceiling": 0.0,
                "mc": 0.0,
                "identity": 0.0,
                "distill": 0.0,
                "me_reg": 0.0,
                "feature_reg": 0.0,
                "expert_reg": 0.0,
                "balance": 0.0,
                "weight": 0.0,
            }

            for frame_idx, frame in enumerate(frames[1:], start=1):
                fa_idx = index_map[frame_idx % args.rate_gop_size]
                with torch.no_grad():
                    teacher_out = teacher_net.forward_one_frame(
                        frame, teacher_dpb, q_index=q_index_p, fa_idx=fa_idx
                    )

                out = p_frame_net.forward_one_frame(
                    frame, dpb, q_index=q_index_p, fa_idx=fa_idx
                )
                recon = out["dpb"]["ref_frame"]
                teacher_recon = teacher_out["dpb"]["ref_frame"]
                mse = F.mse_loss(recon, frame)
                aux = out.get("aux", {})
                mc_loss = F.l1_loss(aux["warpframe"], frame) if "warpframe" in aux else mse * 0
                identity_loss = (
                    F.l1_loss(aux["me_ref"], dpb["ref_frame"]) if "me_ref" in aux else mse * 0
                )
                distill_loss = F.mse_loss(recon, teacher_recon)
                pixel_num = frame.shape[-2] * frame.shape[-1]
                bpp = out["bit"] / (pixel_num * frame.shape[0])
                teacher_bpp = teacher_out["bit"].detach() / (pixel_num * frame.shape[0])
                eps = torch.finfo(bpp.dtype).eps
                log_bpp_distill = F.smooth_l1_loss(
                    torch.log(bpp + eps),
                    torch.log(teacher_bpp + eps),
                )
                bit_ceiling_loss = torch.relu(
                    bpp / (teacher_bpp + eps) - args.bit_ceiling_ratio
                ).pow(2)
                balance = out.get("balance_loss")
                if balance is None:
                    balance = torch.zeros((), dtype=frame.dtype, device=device)
                me_reg = aux.get("hrs_me_delta_reg")
                if me_reg is None:
                    me_reg = torch.zeros((), dtype=frame.dtype, device=device)
                feature_reg = aux.get("hrs_feature_reg")
                if feature_reg is None:
                    feature_reg = torch.zeros((), dtype=frame.dtype, device=device)
                expert_reg = aux.get("hrs_expert_reg")
                if expert_reg is None:
                    expert_reg = torch.zeros((), dtype=frame.dtype, device=device)

                loss = (
                    mse
                    + args.lambda_bpp * bpp
                    + args.lambda_mc * mc_loss
                    + args.lambda_identity * identity_loss
                    + distill_weight * distill_loss
                    + args.lambda_log_bpp_distill * log_bpp_distill
                    + args.lambda_bit_ceiling * bit_ceiling_loss
                    + args.lambda_me_reg * me_reg
                    + args.lambda_feature_reg * feature_reg
                    + args.lambda_expert_reg * expert_reg
                    + args.beta_balance * balance
                )
                frame_weight = args.late_frame_gamma ** (frame_idx - 1)
                weighted_losses.append(loss * frame_weight)

                stats["mse"] += mse.detach() * frame_weight
                stats["bpp"] += bpp.detach() * frame_weight
                stats["teacher_bpp"] += teacher_bpp.detach() * frame_weight
                stats["log_bpp"] += log_bpp_distill.detach() * frame_weight
                stats["bit_ceiling"] += bit_ceiling_loss.detach() * frame_weight
                stats["mc"] += mc_loss.detach() * frame_weight
                stats["identity"] += identity_loss.detach() * frame_weight
                stats["distill"] += distill_loss.detach() * frame_weight
                stats["me_reg"] += me_reg.detach() * frame_weight
                stats["feature_reg"] += feature_reg.detach() * frame_weight
                stats["expert_reg"] += expert_reg.detach() * frame_weight
                stats["balance"] += balance.detach() * frame_weight
                stats["weight"] += frame_weight

                dpb = detach_dpb(out["dpb"]) if args.detach_dpb else out["dpb"]
                teacher_dpb = detach_dpb(teacher_out["dpb"])

            total_loss = torch.stack(weighted_losses).sum() / max(float(stats["weight"]), 1.0)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            step += 1

            if step % args.log_interval == 0 or step == 1:
                denom = max(float(stats["weight"]), 1.0)
                print(
                    f"step {step} q {q_index_p} loss {total_loss.item():.6f} "
                    f"mse {(stats['mse'] / denom).item():.6f} "
                    f"bpp {(stats['bpp'] / denom).item():.6f} "
                    f"tbpp {(stats['teacher_bpp'] / denom).item():.6f} "
                    f"mc {(stats['mc'] / denom).item():.6f} "
                    f"id {(stats['identity'] / denom).item():.6f} "
                    f"distill {(stats['distill'] / denom).item():.6f} "
                    f"logbpp {(stats['log_bpp'] / denom).item():.6f} "
                    f"bitceil {(stats['bit_ceiling'] / denom).item():.6f} "
                    f"me_reg {(stats['me_reg'] / denom).item():.6f} "
                    f"feat_reg {(stats['feature_reg'] / denom).item():.6f} "
                    f"expert_reg {(stats['expert_reg'] / denom).item():.6f} "
                    f"distill_w {distill_weight:.4f} "
                    f"balance {(stats['balance'] / denom).item():.6f}",
                    flush=True,
                )

            if step % args.save_interval == 0:
                save_checkpoint(save_dir / f"{args.checkpoint_prefix}_step{step}.pth.tar",
                                p_frame_net, optimizer, step, epoch, args)

            if step >= target_step:
                break
        if step >= target_step:
            break

    save_checkpoint(save_dir / f"{args.checkpoint_prefix}_latest.pth.tar",
                    p_frame_net, optimizer, step, args.epochs, args)


if __name__ == "__main__":
    main()
