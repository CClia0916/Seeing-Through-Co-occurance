import torch
import torch.nn as nn
import torch.nn.functional as F
from .classification_head import ClassificationHead
from .two_stream_mixer import TwoStreamMixer
from .temporal_encoder import TemporalEncoder
from .relation_encoder import RelationEncoder


def token_split(inputs, fine_ratio=8):
    B, D, T = inputs.shape
    result = torch.zeros_like(inputs[:, :, 0:T // fine_ratio])
    for i in range(T // fine_ratio):
        result[:, :, i:i + 1] = inputs[:, :, (fine_ratio * i):(fine_ratio * i) + 1]
    return result


class TopKTextRouter(nn.Module):
    def __init__(self, stage_dim, text_dim=384, topk=12):
        super().__init__()
        self.query_proj = nn.Linear(stage_dim, text_dim)
        self.query_norm = nn.LayerNorm(text_dim)
        self.topk = topk

    def forward(self, x_stage4, prototypes):
        if prototypes is None:
            return None, None

        x = x_stage4.transpose(1, 2)
        pooled = x.mean(dim=1)
        q = F.normalize(self.query_norm(self.query_proj(pooled)), dim=-1)
        p = F.normalize(prototypes, dim=-1)

        sim = torch.matmul(q, p.t())
        k = min(self.topk, p.size(0))
        topv, topi = torch.topk(sim, k=k, dim=-1)
        selected = prototypes[topi]
        return selected, topv


class StageSemanticAlign(nn.Module):
    def __init__(self, stage_dim, text_dim=384, init_alpha=0.20, init_tau=0.70, init_conf=5.0):
        super().__init__()
        self.query_proj = nn.Linear(stage_dim, text_dim)
        self.query_norm = nn.LayerNorm(text_dim)
        self.proto_norm = nn.LayerNorm(text_dim)
        self.value_proj = nn.Linear(text_dim, stage_dim)
        self.delta_proj = nn.Linear(stage_dim, stage_dim)

        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        self.tau = nn.Parameter(torch.tensor(float(init_tau)))
        self.conf_scale = nn.Parameter(torch.tensor(float(init_conf)))

    def forward(self, x, selected_proto):
        if selected_proto is None:
            return x, None

        x_bt = x.transpose(1, 2)
        q = F.normalize(self.query_norm(self.query_proj(x_bt)), dim=-1)
        p = F.normalize(self.proto_norm(selected_proto), dim=-1)

        tau = self.tau.clamp(min=0.05)
        attn_logits = torch.einsum('btd,bkd->btk', q, p) / tau
        attn = attn_logits.softmax(dim=-1)

        proto_value = self.value_proj(selected_proto)
        z = torch.einsum('btk,bkd->btd', attn, proto_value)

        conf = torch.sigmoid(self.conf_scale.clamp(min=1.0) * attn.max(dim=-1, keepdim=True).values)
        alpha = self.alpha.clamp(min=0.0, max=1.0)
        delta = self.delta_proj(z - x_bt)
        out = x_bt + alpha * conf * delta
        return out.transpose(1, 2).contiguous(), conf


class MultiDimensionalDecoupling(nn.Module):

    def __init__(self, inter_channels, inter_channels_slow, num_block, num_block_spatial, head, mlp_ratio, in_feat_dim,
                 final_embedding_dim, num_classes, fine_ratio=8, inter_tokens=None, head_token=4,
                 k=4, text_dim=384, semantic_topk=12):
        super(MultiDimensionalDecoupling, self).__init__()

        if inter_tokens is None:
            inter_tokens = [256, 128, 64, 32]
        self.inter_tokens = inter_tokens
        self.fine_ratio = fine_ratio
        self.text_dim = text_dim
        self.semantic_topk = min(semantic_topk, num_classes)
        self.semantic_ablation = 'full'

        self.dropout = nn.Dropout()
        self.register_buffer('text_prototypes', None, persistent=False)

        self.TemporalEncoder = TemporalEncoder(
            in_feat_dim=in_feat_dim,
            embed_dims=inter_channels,
            num_head=head,
            mlp_ratio=mlp_ratio,
            norm_layer=nn.LayerNorm,
            num_block=num_block
        )
        self.RelationEncoder = RelationEncoder(
            in_feat_dim=in_feat_dim,
            embed_dims=inter_channels_slow,
            num_head=head_token,
            mlp_ratio=mlp_ratio,
            norm_layer=nn.LayerNorm,
            num_block_spatial=num_block_spatial,
            inter_tokens=inter_tokens,
            k=k
        )
        self.TwoStreamMixer = TwoStreamMixer(
            inter_channels=inter_channels,
            inter_channels_slow=inter_channels_slow,
            embedding_dim=final_embedding_dim,
            fine_ratio=fine_ratio
        )

        self.semantic_router = TopKTextRouter(stage_dim=inter_channels[3], text_dim=text_dim, topk=self.semantic_topk)
        self.semantic_align_stage3 = StageSemanticAlign(stage_dim=inter_channels[2], text_dim=text_dim,
                                                        init_alpha=0.15, init_tau=0.70, init_conf=4.0)
        self.semantic_align_stage4 = StageSemanticAlign(stage_dim=inter_channels[3], text_dim=text_dim,
                                                        init_alpha=0.20, init_tau=0.60, init_conf=5.0)

        self.ClassificationHead = ClassificationHead(num_classes=num_classes, embedding_dim=final_embedding_dim)

    def set_semantic_ablation(self, mode='full'):
        mode = str(mode).lower()
        supported = {'full', 'no_text', 'no_router', 'no_stage3', 'no_stage4', 'no_gate', 'router_only', 'no_align'}
        if mode not in supported:
            raise ValueError(f'Unsupported semantic ablation mode: {mode}')
        self.semantic_ablation = mode

    def set_freq_ablation(self, mode='all'):
        if hasattr(self.TemporalEncoder, 'set_freq_ablation'):
            self.TemporalEncoder.set_freq_ablation(mode)

    def set_text_prototypes(self, phrase_embs):
        if phrase_embs is None:
            self.text_prototypes = None
            return
        if phrase_embs.dim() != 2:
            raise ValueError(f"phrase_embs should be 2D, got shape {tuple(phrase_embs.shape)}")
        if phrase_embs.size(1) != self.text_dim:
            raise ValueError(f"Expected text dim {self.text_dim}, but got {phrase_embs.size(1)}")
        with torch.no_grad():
            self.text_prototypes = F.normalize(phrase_embs.detach().float(), dim=-1)

    def _semantic_align_coarse(self, x_coarse):
        mode = getattr(self, 'semantic_ablation', 'full')
        if (self.text_prototypes is None) or (mode == 'no_text'):
            return x_coarse, [None, None, None, None]

        f1, f2, f3, f4 = x_coarse
        text_bank = self.text_prototypes.to(f4.device)

        if mode == 'no_router':
            selected_proto = text_bank.unsqueeze(0).expand(f4.size(0), -1, -1)
        else:
            selected_proto, _ = self.semantic_router(f4, text_bank)

        if mode == 'router_only':
            return [f1, f2, f3, f4], [None, None, None, None]

        conf3 = None
        conf4 = None

        if mode not in {'no_stage3', 'no_align'}:
            f3, conf3 = self.semantic_align_stage3(f3, selected_proto)
        if mode not in {'no_stage4', 'no_align'}:
            f4, conf4 = self.semantic_align_stage4(f4, selected_proto)

        if mode == 'no_gate':
            conf3 = None
            conf4 = None

        return [f1, f2, f3, f4], [None, None, conf3, conf4]

    def forward(self, inputs):
        inputs = self.dropout(inputs)

        if self.fine_ratio > 1:
            fine_inputs = token_split(inputs, self.fine_ratio)
        else:
            fine_inputs = inputs

        x_fine = self.RelationEncoder(fine_inputs)
        x = self.TemporalEncoder(inputs)

        x, semantic_conf = self._semantic_align_coarse(x)

        concat_feature = self.TwoStreamMixer(x, x_fine, semantic_conf=semantic_conf)

        x, x_hm = self.ClassificationHead(concat_feature)
        return x, x_hm
