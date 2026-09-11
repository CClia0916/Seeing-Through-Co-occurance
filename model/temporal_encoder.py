from torch.nn.init import trunc_normal_
import math
import torch.nn as nn
import torch
import torch.nn.functional as F


class Local_Relational_Block(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.TC1 = nn.Conv1d(hidden_features, hidden_features, 7, 1, 3, bias=True,
                             groups=hidden_features)
        self.TC2 = nn.Conv1d(hidden_features, hidden_features, 5, 1, 2, bias=True,
                             groups=hidden_features)
        self.TC3 = nn.Conv1d(hidden_features, hidden_features, 3, 1, 1, bias=True,
                             groups=hidden_features)

        self.para = torch.nn.Parameter(torch.randn(3), requires_grad=True)

        self.act = act_layer()
        self.linear2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.linear1(x)
        x = x.transpose(1, 2)
        x1 = self.TC1(x)
        x2 = self.TC2(x)
        x3 = self.TC3(x)
        para = torch.sigmoid(self.para)
        x = para[0] * x1 + para[1] * x2 + para[2] * x3
        x = x.transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)
        return x


class PyramidFeatureFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.low_to_mid = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

        self.mid_fusion = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

        self.low_mid_to_high = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

        self.channel_attention = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, low_freq, mid_freq, high_freq):
        low_freq_mapped = self.low_to_mid(low_freq)
        low_freq_mapped = self.norm1(low_freq_mapped + low_freq)

        combined_low_mid = self.mid_fusion(low_freq_mapped + mid_freq)
        combined_low_mid = self.norm2(combined_low_mid + mid_freq)

        combined_low_mid_mapped = self.low_mid_to_high(combined_low_mid)
        combined_low_mid_mapped = self.norm3(combined_low_mid_mapped + combined_low_mid)

        channel_weights = self.channel_attention(combined_low_mid_mapped.mean(dim=1, keepdim=True))
        fused_output = combined_low_mid_mapped * channel_weights + high_freq

        return fused_output


class Global_Relational_Block(nn.Module):
    def __init__(self, dim, num_heads=8, low_freq_ratio=0.2, mid_freq_ratio=0.3, high_freq_ratio=0.5,
                 overlap_ratio=0.1, peak_topk=3, sigma_ratio=0.5, response_lambda=1.0, response_mode="l2"):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = None or head_dim ** -0.5
        self.low_freq_ratio = low_freq_ratio
        self.mid_freq_ratio = mid_freq_ratio
        self.high_freq_ratio = high_freq_ratio
        self.overlap_ratio = overlap_ratio

        self.peak_topk = peak_topk
        self.sigma_ratio = sigma_ratio
        self.response_lambda = response_lambda
        self.response_mode = response_mode

        self.freq_ablation = 'all'

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

        self.pyramid_fusion = PyramidFeatureFusion(dim)

        self.output_layer = nn.Sequential(
            nn.Linear(dim, dim),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def set_freq_ablation(self, mode='all'):
        supported = {'all', 'low_only', 'mid_only', 'high_only', 'no_low', 'no_mid', 'no_high'}
        if mode not in supported:
            raise ValueError(f'Unsupported freq ablation mode: {mode}')
        self.freq_ablation = mode

    def _apply_freq_ablation(self, low_feat, mid_feat, high_feat):
        mode = getattr(self, 'freq_ablation', 'all')

        low_zero = torch.zeros_like(low_feat)
        mid_zero = torch.zeros_like(mid_feat)
        high_zero = torch.zeros_like(high_feat)

        if mode == 'all':
            return low_feat, mid_feat, high_feat
        elif mode == 'low_only':
            return low_feat, mid_zero, high_zero
        elif mode == 'mid_only':
            return low_zero, mid_feat, high_zero
        elif mode == 'high_only':
            return low_zero, mid_zero, high_feat
        elif mode == 'no_low':
            return low_zero, mid_feat, high_feat
        elif mode == 'no_mid':
            return low_feat, mid_zero, high_feat
        elif mode == 'no_high':
            return low_feat, mid_feat, high_zero
        else:
            return low_feat, mid_feat, high_feat

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def apply_attention(self, x_input, B, N, C):
        q = self.q(x_input).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x_input).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x_input).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out

    @staticmethod
    def align_tensor(x, target_T, mode='linear'):
        B, Tx, D = x.shape

        if mode == 'linear':
            x = x.permute(0, 2, 1)
            x = F.interpolate(x, size=target_T, mode='linear')
            return x.permute(0, 2, 1)
        elif mode == 'adaptive_avg':
            return F.adaptive_avg_pool1d(x, target_T)
        elif mode == 'adaptive_max':
            return F.adaptive_max_pool1d(x, target_T)

    def _build_gaussian_band_mask(self, energy_spec, left, right, total_freqs):
        device = energy_spec.device
        dtype = energy_spec.dtype
        mask = torch.zeros(energy_spec.size(0), total_freqs, device=device, dtype=dtype)
        if right <= left:
            return mask

        band_energy = energy_spec[:, left:right]
        width = right - left
        k = max(1, min(self.peak_topk, width))

        topk_val, topk_idx = torch.topk(band_energy, k=k, dim=-1)
        topk_freq = topk_idx.to(dtype) + float(left)
        denom = topk_val.sum(dim=-1, keepdim=True) + 1e-6
        mu = (topk_freq * topk_val).sum(dim=-1, keepdim=True) / denom

        left_span = mu - float(left)
        right_span = float(right - 1) - mu
        sigma = torch.maximum(left_span, right_span)
        sigma = torch.clamp(sigma * self.sigma_ratio, min=1.0)

        freq_idx = torch.arange(total_freqs, device=device, dtype=dtype).view(1, -1)
        gaussian = torch.exp(-0.5 * ((freq_idx - mu) / (sigma + 1e-6)) ** 2)

        band_binary = torch.zeros(1, total_freqs, device=device, dtype=dtype)
        band_binary[:, left:right] = 1.0
        mask = gaussian * band_binary

        band_gain = band_energy.sum(dim=-1, keepdim=True) / (energy_spec.sum(dim=-1, keepdim=True) + 1e-6)
        mask = mask * band_gain
        return mask

    def _temporal_reweight(self, x_band):
        if self.response_mode == "l1":
            response = x_band.abs().mean(dim=-1, keepdim=True)
        else:
            response = x_band.pow(2).mean(dim=-1, keepdim=True)

        response = response / (response.sum(dim=1, keepdim=True) + 1e-6)
        x_band = x_band * (1.0 + self.response_lambda * response)
        return x_band

    def forward(self, x):
        B, N, C = x.shape

        x1 = x.transpose(1, 2)
        x_freq = torch.fft.rfft(x1, dim=-1)

        total_freqs = x_freq.shape[-1]
        overlap = int(total_freqs * self.overlap_ratio)
        low_freq_end = int(total_freqs * self.low_freq_ratio)
        mid_freq_end = low_freq_end + int(total_freqs * self.mid_freq_ratio)

        low_left, low_right = 0, min(total_freqs, low_freq_end + overlap)
        mid_left, mid_right = max(0, low_freq_end - overlap), min(total_freqs, mid_freq_end + overlap)
        high_left, high_right = max(0, mid_freq_end - overlap), total_freqs

        energy_spec = (x_freq.abs() ** 2).mean(dim=1)

        low_mask = self._build_gaussian_band_mask(energy_spec, low_left, low_right, total_freqs)
        mid_mask = self._build_gaussian_band_mask(energy_spec, mid_left, mid_right, total_freqs)
        high_mask = self._build_gaussian_band_mask(energy_spec, high_left, high_right, total_freqs)

        low_freq = x_freq * low_mask.unsqueeze(1)
        mid_freq = x_freq * mid_mask.unsqueeze(1)
        high_freq = x_freq * high_mask.unsqueeze(1)

        low_freq_time = torch.fft.irfft(low_freq, n=N, dim=-1).transpose(1, 2)
        mid_freq_time = torch.fft.irfft(mid_freq, n=N, dim=-1).transpose(1, 2)
        high_freq_time = torch.fft.irfft(high_freq, n=N, dim=-1).transpose(1, 2)

        low_freq_out = self._temporal_reweight(low_freq_time)
        mid_freq_out = self._temporal_reweight(mid_freq_time)
        high_freq_out = self._temporal_reweight(high_freq_time)

        low_freq_out, mid_freq_out, high_freq_out = self._apply_freq_ablation(
            low_freq_out, mid_freq_out, high_freq_out
        )

        freq_out = low_freq_out + mid_freq_out + high_freq_out
        freq_out = self.norm1(freq_out)

        time_out = self.apply_attention(x, B, N, C)
        time_out = self.norm2(time_out)

        combined_out = self.fusion(torch.cat([freq_out, time_out], dim=-1))
        output = self.output_layer(combined_out)
        return output


class GLRBlock(nn.Module):
    """
    Global Local Relational Block
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.Global_Relational_Block = Global_Relational_Block(
            dim, num_heads=num_heads)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.Local_Relational_Block = Local_Relational_Block(in_features=dim, hidden_features=mlp_hidden_dim,
                                                             act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def set_freq_ablation(self, mode='all'):
        self.Global_Relational_Block.set_freq_ablation(mode)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.Global_Relational_Block(self.norm1(x))
        x = x + self.Local_Relational_Block(self.norm2(x))
        return x


class Temporal_Merging_Block(nn.Module):
    """
    Temporal_Merging_Block
    """

    def __init__(self, kernel_size=3, stride=1, in_chans=1024, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride,
                              padding=(kernel_size // 2))
        self.proj2 = nn.Conv1d(in_chans, embed_dim, kernel_size, stride, padding=(kernel_size // 2))
        self.proj3 = nn.Conv1d(in_chans, embed_dim, kernel_size, stride, padding=(kernel_size // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x) + self.proj2(x) + self.proj3(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class tAPE(nn.Module):
    def __init__(self, d_model, dropout=0.1, scale_factor=1.0):
        super(tAPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.dim = d_model
        self.scale_factor = scale_factor

    def forward(self, x):
        B, N, C = x.shape
        pe = torch.zeros(N, self.dim)
        position = torch.arange(0, N, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim, 2).float() * (-math.log(10000.0) / self.dim))

        pe[:, 0::2] = torch.sin((position * div_term) * (self.dim / N))
        pe[:, 1::2] = torch.cos((position * div_term) * (self.dim / N))
        pe = self.scale_factor * pe.unsqueeze(0)
        x = x + pe.to(x.device)
        return self.dropout(x)


class TemporalEncoder(nn.Module):
    def __init__(self, in_feat_dim=1024, embed_dims=[256, 384, 576, 864],
                 num_head=8, mlp_ratio=8, norm_layer=nn.LayerNorm,
                 num_block=[3, 3, 3, 3]):
        super().__init__()

        self.freq_ablation = 'all'

        self.linear2 = nn.Sequential(nn.Linear(embed_dims[1] * 2, embed_dims[1]))
        self.linear3 = nn.Sequential(nn.Linear(embed_dims[2] * 2, embed_dims[2]))
        self.linear4 = nn.Sequential(nn.Linear(embed_dims[3] * 2, embed_dims[3]))

        # Stage 1
        self.Temporal_Merging_Block1 = Temporal_Merging_Block(kernel_size=3, stride=1, in_chans=in_feat_dim,
                                                              embed_dim=embed_dims[0])
        self.block1 = nn.ModuleList([GLRBlock(
            dim=embed_dims[0], num_heads=num_head, mlp_ratio=mlp_ratio, norm_layer=norm_layer)
            for i in range(num_block[0])])
        self.norm1 = norm_layer(embed_dims[0])

        # Stage 2
        self.Temporal_Merging_Block2 = Temporal_Merging_Block(kernel_size=3, stride=2, in_chans=embed_dims[0],
                                                              embed_dim=embed_dims[1])
        self.Temporal_Merging_Block2_ = Temporal_Merging_Block(kernel_size=3, stride=2, in_chans=embed_dims[0],
                                                               embed_dim=embed_dims[1])
        self.block2 = nn.ModuleList([GLRBlock(
            dim=embed_dims[1], num_heads=num_head, mlp_ratio=mlp_ratio, norm_layer=norm_layer)
            for i in range(num_block[1])])
        self.norm2 = norm_layer(embed_dims[1])

        # Stage 3
        self.Temporal_Merging_Block3 = Temporal_Merging_Block(kernel_size=3, stride=4, in_chans=embed_dims[0],
                                                              embed_dim=embed_dims[2])
        self.Temporal_Merging_Block3_ = Temporal_Merging_Block(kernel_size=3, stride=2, in_chans=embed_dims[1],
                                                               embed_dim=embed_dims[2])
        self.block3 = nn.ModuleList([GLRBlock(
            dim=embed_dims[2], num_heads=num_head, mlp_ratio=mlp_ratio, norm_layer=norm_layer)
            for i in range(num_block[2])])
        self.norm3 = norm_layer(embed_dims[2])

        # Stage 4
        self.Temporal_Merging_Block4 = Temporal_Merging_Block(kernel_size=3, stride=8, in_chans=embed_dims[0],
                                                              embed_dim=embed_dims[3])
        self.Temporal_Merging_Block4_ = Temporal_Merging_Block(kernel_size=3, stride=2, in_chans=embed_dims[2],
                                                               embed_dim=embed_dims[3])
        self.block4 = nn.ModuleList([GLRBlock(
            dim=embed_dims[3], num_heads=num_head, mlp_ratio=mlp_ratio, norm_layer=norm_layer)
            for i in range(num_block[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)
        self.set_freq_ablation(self.freq_ablation)

    def set_freq_ablation(self, mode='all'):
        supported = {'all', 'low_only', 'mid_only', 'high_only', 'no_low', 'no_mid', 'no_high'}
        if mode not in supported:
            raise ValueError(f'Unsupported freq ablation mode: {mode}')
        self.freq_ablation = mode

        for blk in self.block1:
            blk.set_freq_ablation(mode)
        for blk in self.block2:
            blk.set_freq_ablation(mode)
        for blk in self.block3:
            blk.set_freq_ablation(mode)
        for blk in self.block4:
            blk.set_freq_ablation(mode)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_init_emb(self):
        self.Temporal_Merging_Block1.requires_grad = False

    def forward(self, x):
        outs = []

        # stage 1
        s = x
        x = self.Temporal_Merging_Block1(s)
        x1 = x.permute(0, 2, 1).contiguous()
        outs.append(x1)

        # stage 2
        x = self.Temporal_Merging_Block2(x1)
        for i, blk in enumerate(self.block2):
            x = blk(x)
        x = self.norm2(x)
        x2 = x.permute(0, 2, 1).contiguous()
        outs.append(x2)

        # stage 3
        x = self.Temporal_Merging_Block3(x1)
        y = self.Temporal_Merging_Block3_(x2)
        x = torch.cat((x, y), -1)
        x = self.linear3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x)
        x = self.norm3(x)
        x3 = x.permute(0, 2, 1).contiguous()
        outs.append(x3)

        # stage 4
        x = self.Temporal_Merging_Block4(x1)
        y = self.Temporal_Merging_Block4_(x3)
        x = torch.cat((x, y), -1)
        x = self.linear4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x)
        x = self.norm4(x)
        x = x.permute(0, 2, 1).contiguous()
        outs.append(x)

        return outs
