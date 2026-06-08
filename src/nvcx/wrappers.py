import torch
from torch import nn

from ..models.layers import DepthConvBlock


def zero_init_last_conv(module):
    for layer in reversed(list(module.modules())):
        if isinstance(layer, nn.Conv2d):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            return


def make_q_map(x, q_index, qp_num=64):
    if isinstance(q_index, torch.Tensor):
        q = q_index.to(device=x.device, dtype=x.dtype).flatten()
        if q.numel() == 0:
            q = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        elif q.numel() == 1:
            q = q.expand(x.shape[0])
        elif q.numel() != x.shape[0]:
            q = q[:1].expand(x.shape[0])
    elif isinstance(q_index, (list, tuple)):
        if len(q_index) == 0:
            q = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        else:
            q = torch.tensor(q_index, device=x.device, dtype=x.dtype).flatten()
            if q.numel() == 1:
                q = q.expand(x.shape[0])
            elif q.numel() != x.shape[0]:
                q = q[:1].expand(x.shape[0])
    else:
        q = torch.full((x.shape[0],), float(q_index), device=x.device, dtype=x.dtype)
    q = q.clamp(0, qp_num - 1) / float(qp_num - 1)
    return q.view(-1, 1, 1, 1).expand(-1, 1, x.shape[-2], x.shape[-1])


class NeuralPreProcessor(nn.Module):
    def __init__(self, channels=32, qp_num=64, max_alpha=0.20, inplace=False):
        super().__init__()
        self.qp_num = qp_num
        self.max_alpha = max_alpha
        self.stem = nn.Sequential(
            nn.Conv2d(4, channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=inplace),
        )
        self.body = nn.Sequential(
            DepthConvBlock(channels, channels, inplace=inplace),
            DepthConvBlock(channels, channels, inplace=inplace),
            DepthConvBlock(channels, channels, inplace=inplace),
            DepthConvBlock(channels, channels, inplace=inplace),
        )
        self.head = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x, q_index):
        q_map = make_q_map(x, q_index, self.qp_num)
        delta = torch.tanh(self.head(self.body(self.stem(torch.cat((x, q_map), dim=1)))))
        alpha = 0.08 + (self.max_alpha - 0.08) * (1.0 - q_map.mean(dim=(2, 3), keepdim=True))
        return (x + alpha * delta).clamp(0, 1), {"delta_pre": delta, "q_map": q_map}


class NeuralPostProcessor(nn.Module):
    def __init__(self, channels=64, qp_num=64, residual_scale=0.1, inplace=False):
        super().__init__()
        self.qp_num = qp_num
        self.residual_scale = residual_scale
        self.enc = nn.Sequential(
            nn.Conv2d(3 + 48 + 1, channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=inplace),
            DepthConvBlock(channels, channels, inplace=inplace),
            DepthConvBlock(channels, channels, inplace=inplace),
        )
        self.refine = nn.Sequential(
            DepthConvBlock(channels, channels, inplace=inplace),
            DepthConvBlock(channels, channels, inplace=inplace),
            nn.Conv2d(channels, 3, 3, padding=1),
        )
        zero_init_last_conv(self.refine)

    def forward(self, x_hat_core, core_feature, q_index, temporal_feature=None):
        q_map = make_q_map(x_hat_core, q_index, self.qp_num)
        feat = self.enc(torch.cat((x_hat_core, core_feature, q_map), dim=1))
        if temporal_feature is not None:
            feat = feat + temporal_feature
        residual = torch.tanh(self.refine(feat))
        alpha = self.residual_scale * (0.5 + 0.5 * (1.0 - q_map.mean(dim=(2, 3), keepdim=True)))
        return (x_hat_core + alpha * residual).clamp(0, 1), {"post_feature": feat}


class NeuralWrapper(nn.Module):
    def __init__(
        self,
        core_codec,
        qp_num=64,
        inplace=False,
        use_pre=True,
        use_post=True,
        post_as_ref=True,
    ):
        super().__init__()
        self.core = core_codec
        self.pre = NeuralPreProcessor(qp_num=qp_num, inplace=inplace)
        self.post = NeuralPostProcessor(qp_num=qp_num, inplace=inplace)
        self.use_pre = use_pre
        self.use_post = use_post
        self.post_as_ref = post_as_ref

    @staticmethod
    def _write_display_to_dpb(dpb, display_frame):
        updated = dict(dpb)
        updated["ref_frame"] = display_frame
        return updated

    def forward_one_frame(self, x, dpb, q_index, fa_idx, wrapper_state=None):
        if self.use_pre:
            x_pre, pre_aux = self.pre(x, q_index)
        else:
            x_pre, pre_aux = x, {}
        core_out = self.core.forward_one_frame(x_pre, dpb, q_index=q_index, fa_idx=fa_idx)
        core_dpb = core_out["dpb"]
        temporal_feature = None
        if wrapper_state is not None:
            temporal_feature = wrapper_state.get("post_feature")
        if self.use_post:
            x_tilde, post_aux = self.post(
                core_dpb["ref_frame"],
                core_dpb["ref_feature"],
                q_index,
                temporal_feature=temporal_feature,
            )
        else:
            x_tilde, post_aux = core_dpb["ref_frame"], {}
        wrapper_state = {}
        if "post_feature" in post_aux:
            wrapper_state["post_feature"] = post_aux["post_feature"]
        if "delta_pre" in pre_aux:
            wrapper_state["pre_delta"] = pre_aux["delta_pre"]
        updated_dpb = (
            self._write_display_to_dpb(core_dpb, x_tilde)
            if self.use_post and self.post_as_ref else core_dpb
        )
        return {
            "x_tilde": x_tilde,
            "x_hat_core": core_dpb["ref_frame"],
            "dpb": updated_dpb,
            "bit": core_out["bit"],
            "wrapper_state": wrapper_state,
            "pre_aux": pre_aux,
            "post_aux": post_aux,
        }

    def encode(self, x, dpb, q_index, fa_idx, sps_id=0, output_file=None):
        if output_file is not None:
            if self.use_pre:
                x_pre, pre_aux = self.pre(x, q_index)
            else:
                x_pre, pre_aux = x, {}
            encoded = self.core.encode(x_pre, dpb, q_index, fa_idx, sps_id, output_file)
            if self.use_post:
                display, post_aux = self.post(
                    encoded["dpb"]["ref_frame"],
                    encoded["dpb"]["ref_feature"],
                    q_index,
                )
            else:
                display, post_aux = encoded["dpb"]["ref_frame"], {}
            encoded["display_frame"] = display
            if self.use_post and self.post_as_ref:
                encoded["dpb"] = self._write_display_to_dpb(encoded["dpb"], display)
            encoded["wrapper_state"] = {}
            if "post_feature" in post_aux:
                encoded["wrapper_state"]["post_feature"] = post_aux["post_feature"]
            if "delta_pre" in pre_aux:
                encoded["wrapper_state"]["pre_delta"] = pre_aux["delta_pre"]
            return encoded

        encoded = self.forward_one_frame(x, dpb, q_index, fa_idx)
        return {
            "dpb": encoded["dpb"],
            "bit": encoded["bit"].item(),
            "display_frame": encoded["x_tilde"],
            "wrapper_state": encoded["wrapper_state"],
        }

    def decompress(self, bit_stream, dpb, sps):
        decoded = self.core.decompress(bit_stream, dpb, sps)
        if self.use_post:
            display, post_aux = self.post(
                decoded["dpb"]["ref_frame"],
                decoded["dpb"]["ref_feature"],
                sps["qp"],
            )
        else:
            display, post_aux = decoded["dpb"]["ref_frame"], {}
        decoded["display_frame"] = display
        if self.use_post and self.post_as_ref:
            decoded["dpb"] = self._write_display_to_dpb(decoded["dpb"], display)
        decoded["wrapper_state"] = {}
        if "post_feature" in post_aux:
            decoded["wrapper_state"]["post_feature"] = post_aux["post_feature"]
        return decoded

    def update(self, force=False):
        return self.core.update(force=force)
