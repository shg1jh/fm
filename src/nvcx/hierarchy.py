from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ReferenceState:
    feat1: torch.Tensor
    feat2: torch.Tensor
    feat3: torch.Tensor
    lowrank3: torch.Tensor
    q_index: int
    fa_idx: int = 0
    age: int = 0


@dataclass
class HierarchyOutput:
    me_ref: torch.Tensor
    ref_pyramid: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    route_logits: Optional[torch.Tensor] = None
    route_topk: Optional[torch.Tensor] = None
    balance_loss: Optional[torch.Tensor] = None
    me_delta_reg: Optional[torch.Tensor] = None
    feature_reg: Optional[torch.Tensor] = None
    expert_reg: Optional[torch.Tensor] = None
    gate_values: Optional[torch.Tensor] = None


def _zero_init_last_conv(module):
    for layer in reversed(list(module.modules())):
        if isinstance(layer, nn.Conv2d):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            return


def _as_index_tensor(value, batch_size, device, max_value):
    if isinstance(value, torch.Tensor):
        value = value.to(device=device, dtype=torch.long).flatten()
        if value.numel() == 0:
            value = torch.zeros(batch_size, device=device, dtype=torch.long)
        elif value.numel() == 1:
            value = value.expand(batch_size)
        elif value.numel() != batch_size:
            value = value[:1].expand(batch_size)
    elif isinstance(value, (list, tuple)):
        if len(value) == 0:
            value = torch.zeros(batch_size, device=device, dtype=torch.long)
        else:
            value = torch.tensor(value, device=device, dtype=torch.long).flatten()
            if value.numel() == 1:
                value = value.expand(batch_size)
            elif value.numel() != batch_size:
                value = value[:1].expand(batch_size)
    else:
        value = torch.full((batch_size,), int(value), device=device, dtype=torch.long)
    return value.clamp_(0, max_value)


class HierarchicalContextModulation(nn.Module):
    def __init__(
        self,
        channels=(48, 64, 96),
        num_experts=3,
        rank_ratio=4,
        qp_num=64,
        fa_num=4,
        max_age=8,
        me_delta_scale=0.02,
        max_alpha_hist=0.05,
        max_alpha_expert=0.03,
        gate_init=6.0,
        inplace=False,
    ):
        super().__init__()
        c1, c2, c3 = channels
        rank = max(c1 // rank_ratio, 1)

        self.num_experts = num_experts
        self.max_age = max_age
        self.me_delta_scale = me_delta_scale
        self.max_alpha_hist = max_alpha_hist
        self.max_alpha_expert = max_alpha_expert
        self.global_strength = nn.Parameter(torch.full((3,), float(gate_init)))
        self.q_strength = nn.Embedding(qp_num, 3)
        nn.init.zeros_(self.q_strength.weight)

        self.hist_fusion1 = nn.Sequential(
            nn.Conv2d(c1 * 2, c1, 1),
            nn.LeakyReLU(0.1, inplace=inplace),
            nn.Conv2d(c1, c1, 1),
        )
        self.hist_fusion2 = nn.Sequential(
            nn.Conv2d(c2 * 2, c2, 1),
            nn.LeakyReLU(0.1, inplace=inplace),
            nn.Conv2d(c2, c2, 1),
        )
        self.hist_fusion3 = nn.Sequential(
            nn.Conv2d(c3 * 2, c3, 1),
            nn.LeakyReLU(0.1, inplace=inplace),
            nn.Conv2d(c3, c3, 1),
        )
        _zero_init_last_conv(self.hist_fusion1)
        _zero_init_last_conv(self.hist_fusion2)
        _zero_init_last_conv(self.hist_fusion3)

        self.proj_main = nn.Conv2d(c1, c1, 1)
        self.proj_gate = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=inplace),
            nn.Conv2d(c1, c1, 1),
        )

        self.q_embed = nn.Embedding(qp_num, c1)
        self.age_embed = nn.Embedding(max_age + 1, c1)
        self.fa_embed = nn.Embedding(fa_num, c1)
        self.router = nn.Linear(c1, num_experts)

        self.expert_m = nn.ModuleList([nn.Conv2d(c1, rank, 1) for _ in range(num_experts)])
        self.expert_g = nn.ModuleList([nn.Conv2d(c1, rank, 1) for _ in range(num_experts)])
        self.expert_o = nn.ModuleList([nn.Conv2d(rank, c1, 1) for _ in range(num_experts)])
        for expert in self.expert_o:
            nn.init.zeros_(expert.weight)
            if expert.bias is not None:
                nn.init.zeros_(expert.bias)

        self.to_me_delta = nn.Sequential(
            nn.Conv2d(c1, c1 // 2, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=inplace),
            nn.Conv2d(c1 // 2, 3, 1),
        )
        _zero_init_last_conv(self.to_me_delta)

    @staticmethod
    def _resize_like(x, target):
        if x.shape[-2:] == target.shape[-2:]:
            return x
        return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)

    def _mean_history(self, states: List[ReferenceState], attr: str, target):
        if len(states) <= 1:
            return torch.zeros_like(target)
        items = []
        for state in states[1:]:
            value = getattr(state, attr)
            items.append(self._resize_like(value, target))
        return torch.stack(items, dim=0).mean(dim=0)

    def forward(
        self,
        ref_states: List[ReferenceState],
        q_index,
        fa_idx=0,
        mode="encode",
        route_id: Optional[torch.Tensor] = None,
    ) -> HierarchyOutput:
        if len(ref_states) == 0:
            raise ValueError("HierarchicalContextModulation requires at least one reference state.")

        base1 = ref_states[0].feat1
        base2 = ref_states[0].feat2
        base3 = ref_states[0].feat3
        base_lowrank = ref_states[0].lowrank3

        batch_size = base1.shape[0]
        q_ids = _as_index_tensor(q_index, batch_size, base1.device,
                                 self.q_embed.num_embeddings - 1)
        fa_ids = _as_index_tensor(fa_idx, batch_size, base1.device,
                                  self.fa_embed.num_embeddings - 1)
        age_ids = _as_index_tensor(
            ref_states[0].age, batch_size, base1.device, self.age_embed.num_embeddings - 1
        )
        gates = torch.sigmoid(self.global_strength.view(1, 3) + self.q_strength(q_ids))
        alpha_hist = (self.max_alpha_hist * gates[:, 0]).view(batch_size, 1, 1, 1)
        alpha_expert = (self.max_alpha_expert * gates[:, 1]).view(batch_size, 1, 1, 1)
        alpha_me = (self.me_delta_scale * gates[:, 2]).view(batch_size, 1, 1, 1)

        hist1 = self._mean_history(ref_states, "feat1", base1)
        hist2 = self._mean_history(ref_states, "feat2", base2)
        hist3 = self._mean_history(ref_states, "feat3", base3)

        hist_delta1 = self.hist_fusion1(torch.cat((base1, hist1), dim=1))
        hist_delta2 = self.hist_fusion2(torch.cat((base2, hist2), dim=1))
        hist_delta3 = self.hist_fusion3(torch.cat((base3, hist3), dim=1))
        feat1 = base1 + alpha_hist * hist_delta1
        feat2 = base2 + alpha_hist * hist_delta2
        feat3 = base3 + alpha_hist * hist_delta3

        main = self.proj_main(feat1)
        gate_feat = self.proj_gate(feat1)

        pooled = F.adaptive_avg_pool2d(main, 1).flatten(1)
        pooled = pooled + self.q_embed(q_ids) + self.fa_embed(fa_ids) + self.age_embed(age_ids)
        logits = self.router(pooled)

        if route_id is None:
            top1 = logits.argmax(dim=1)
        else:
            top1 = _as_index_tensor(route_id, batch_size, feat1.device, self.num_experts - 1)

        expert_sum = torch.zeros_like(main)
        for idx in range(self.num_experts):
            active = (top1 == idx).to(dtype=main.dtype).view(-1, 1, 1, 1)
            low_m = self.expert_m[idx](main)
            low_g = torch.sigmoid(self.expert_g[idx](gate_feat))
            expert_sum = expert_sum + active * self.expert_o[idx](low_m * low_g)
        expert_delta = alpha_expert * expert_sum
        feat1 = feat1 + expert_delta

        me_delta = torch.tanh(self.to_me_delta(feat1))
        me_ref = (base_lowrank + alpha_me * me_delta).clamp(0, 1)

        feature_reg = (
            F.smooth_l1_loss(feat1, base1.detach())
            + F.smooth_l1_loss(feat2, base2.detach())
            + F.smooth_l1_loss(feat3, base3.detach())
        )
        me_delta_reg = F.smooth_l1_loss(me_ref, base_lowrank.detach())
        expert_reg = torch.mean(torch.abs(expert_delta))

        balance_loss = None
        if self.training:
            probs = F.softmax(logits, dim=1)
            one_hot = F.one_hot(top1, self.num_experts).to(dtype=probs.dtype)
            balance_loss = self.num_experts * (probs * one_hot).sum(dim=1).mean()

        return HierarchyOutput(
            me_ref=me_ref,
            ref_pyramid=(feat1, feat2, feat3),
            route_logits=logits,
            route_topk=top1,
            balance_loss=balance_loss,
            me_delta_reg=me_delta_reg,
            feature_reg=feature_reg,
            expert_reg=expert_reg,
            gate_values=gates.mean(dim=0),
        )
