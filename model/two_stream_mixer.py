import torch
import torch.nn as nn
import torch.nn.functional as F


class linear_layer(nn.Module):
    #
    def __init__(self, input_dim=2048, embed_dim=512):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.proj(x)
        return x


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None):
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class TwoStreamMixer(nn.Module):
    # inter_channels = [96, 192, 384, 768]
    # inter_channels_slow = [256, 384, 576, 864]
    # embedding_dim = final_embedding_dim = 512
    def __init__(self, inter_channels, inter_channels_slow, embedding_dim, fine_ratio):
        super().__init__()
        self.fine_ratio = fine_ratio  # 8
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = inter_channels
        c1_in_channels_slow, c2_in_channels_slow, c3_in_channels_slow, c4_in_channels_slow = inter_channels_slow

        self.linear_f4_fine = linear_layer(input_dim=c4_in_channels_slow, embed_dim=embedding_dim)
        self.linear_f3_fine = linear_layer(input_dim=c3_in_channels_slow, embed_dim=embedding_dim)
        self.linear_f2_fine = linear_layer(input_dim=c2_in_channels_slow, embed_dim=embedding_dim)
        self.linear_f1_fine = linear_layer(input_dim=c1_in_channels_slow, embed_dim=embedding_dim)

        self.linear_f4 = linear_layer(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_f3 = linear_layer(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_f2 = linear_layer(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_f1 = linear_layer(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_f4_cat = linear_layer(input_dim=embedding_dim * 1, embed_dim=embedding_dim)
        self.linear_f3_cat = linear_layer(input_dim=embedding_dim * 1, embed_dim=embedding_dim)
        self.linear_f2_cat = linear_layer(input_dim=embedding_dim * 1, embed_dim=embedding_dim)
        self.linear_f1_cat = linear_layer(input_dim=embedding_dim * 1, embed_dim=embedding_dim)

        self.linear_f4_coarse = linear_layer(input_dim=embedding_dim, embed_dim=embedding_dim)
        self.linear_f3_coarse = linear_layer(input_dim=embedding_dim, embed_dim=embedding_dim)
        self.linear_f2_coarse = linear_layer(input_dim=embedding_dim, embed_dim=embedding_dim)
        self.linear_f1_coarse = linear_layer(input_dim=embedding_dim, embed_dim=embedding_dim)

        self.linear1 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1)
        self.linear2 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1)
        self.linear3 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1)

        # 只在 stage3 / stage4 上做很轻的语义尺度仲裁
        self.stage3_gate_scale = nn.Parameter(torch.tensor(0.20))
        self.stage4_gate_scale = nn.Parameter(torch.tensor(0.25))

    @staticmethod
    def _apply_semantic_gate(fine_feat, conf_map, gate_scale):
        if conf_map is None:
            return fine_feat

        # conf_map: [B, T, 1] -> [B, 1, T]
        conf = conf_map.permute(0, 2, 1).contiguous()
        if conf.size(-1) != fine_feat.size(-1):
            conf = resize(conf, size=fine_feat.size(-1), mode='linear', align_corners=False)

        scale = 1.0 + gate_scale.clamp(min=0.0, max=1.0) * conf
        return fine_feat * scale

    def forward(self, x_coarse, x_fine, semantic_conf=None):
        f1, f2, f3, f4 = x_coarse
        f1_fine, f2_fine, f3_fine, f4_fine = x_fine

        conf3 = None
        conf4 = None
        if semantic_conf is not None:
            if len(semantic_conf) > 2:
                conf3 = semantic_conf[2]
            if len(semantic_conf) > 3:
                conf4 = semantic_conf[3]

        # =================================================
        # Temporal Scale Mixer Module For Add Stream ======
        _f4_fine = self.linear_f4_fine(f4_fine).permute(0, 2, 1)
        _f4_fine = resize(_f4_fine, scale_factor=self.fine_ratio, mode='linear', align_corners=False)
        _f4_fine = self._apply_semantic_gate(_f4_fine, conf4, self.stage4_gate_scale)

        _f3_fine = self.linear_f3_fine(f3_fine).permute(0, 2, 1)
        _f3_fine = resize(_f3_fine, scale_factor=self.fine_ratio, mode='linear', align_corners=False)
        _f3_fine = self._apply_semantic_gate(_f3_fine, conf3, self.stage3_gate_scale)

        _f2_fine = self.linear_f2_fine(f2_fine).permute(0, 2, 1)
        _f2_fine = resize(_f2_fine, scale_factor=self.fine_ratio, mode='linear', align_corners=False)

        _f1_fine = self.linear_f1_fine(f1_fine).permute(0, 2, 1)
        _f1_fine = resize(_f1_fine, scale_factor=self.fine_ratio, mode='linear', align_corners=False)
        # =================================================

        # =================================================
        # Temporal Scale Mixer Module ======
        _f4 = self.linear_f4(f4).permute(0, 2, 1)
        _f4 = self.linear_f4_cat(_f4 + _f4_fine).permute(0, 2, 1)
        _f4_nextIn = resize(_f4, scale_factor=2, mode='linear', align_corners=False)
        _f4 = self.linear_f4_coarse(_f4).permute(0, 2, 1)
        _f4 = resize(_f4, size=f1.size()[2:], mode='linear', align_corners=False)

        _f3 = self.linear_f3(f3).permute(0, 2, 1)
        _f3 = self.linear_f3_cat(_f3 + _f3_fine).permute(0, 2, 1)
        _f3_nextIn = resize(_f3, scale_factor=2, mode='linear', align_corners=False)
        _f3 = self.linear_f3_coarse(_f3).permute(0, 2, 1) + self.linear3(_f4_nextIn)
        _f3 = resize(_f3, size=f1.size()[2:], mode='linear', align_corners=False)

        _f2 = self.linear_f2(f2).permute(0, 2, 1)
        _f2 = self.linear_f2_cat(_f2 + _f2_fine).permute(0, 2, 1)
        _f2_nextIn = resize(_f2, scale_factor=2, mode='linear', align_corners=False)
        _f2 = self.linear_f2_coarse(_f2).permute(0, 2, 1) + self.linear2(_f3_nextIn)
        _f2 = resize(_f2, size=f1.size()[2:], mode='linear', align_corners=False)

        _f1 = self.linear_f1(f1).permute(0, 2, 1)
        _f1 = self.linear_f1_cat(_f1 + _f1_fine).permute(0, 2, 1)
        _f1 = self.linear_f1_coarse(_f1).permute(0, 2, 1) + self.linear1(_f2_nextIn)
        # =================================================

        concat_feature = torch.cat([_f4, _f3, _f2, _f1], dim=1)
        return concat_feature
