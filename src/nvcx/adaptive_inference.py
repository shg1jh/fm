import torch
import torch.nn.functional as F

from ..models.block_mc import block_mc_func


class AdaptiveMVScaleSelector:
    def __init__(self, candidates=(1.0, 1.25, 1.5, 2.0, 3.0, 4.0), eta_mv=0.002):
        self.candidates = tuple(float(v) for v in candidates)
        self.eta_mv = eta_mv

    @torch.no_grad()
    def select(self, x, me_ref, ref_frame, flow_net, warp_fn=None):
        warp_fn = block_mc_func if warp_fn is None else warp_fn
        height, width = x.shape[-2:]
        best = None
        for scale in self.candidates:
            if scale == 1.0:
                x_scaled = x
                ref_scaled = me_ref
            else:
                scaled_size = (
                    max(int(round(height / scale)), 8),
                    max(int(round(width / scale)), 8),
                )
                x_scaled = F.interpolate(
                    x, size=scaled_size, mode="bilinear", align_corners=False
                )
                ref_scaled = F.interpolate(
                    me_ref, size=scaled_size, mode="bilinear", align_corners=False
                )

            flow_scaled = flow_net(x_scaled, ref_scaled)
            if flow_scaled.shape[-2:] != (height, width):
                flow_scaled = F.interpolate(
                    flow_scaled, size=(height, width), mode="bilinear", align_corners=False
                )
            flow_comp = flow_scaled * scale
            pred = warp_fn(ref_frame, flow_comp)
            warp_loss = (x - pred).pow(2).mean()
            mv_proxy_rate = flow_scaled.abs().mean()
            score = warp_loss + self.eta_mv * mv_proxy_rate

            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "scale": scale,
                    "flow_scaled": flow_scaled,
                    "flow_comp": flow_comp,
                }
        return best
