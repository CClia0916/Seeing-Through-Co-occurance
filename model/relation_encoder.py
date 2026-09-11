from torch.nn.init import trunc_normal_
import math
import torch.nn as nn

"""
这里在MLP模块中直接使用两个linear层，舍弃了卷积层（该结构源自Swin）
"""


class Global_Relational_Block_Win(nn.Module):
    def __init__(self, inter_token, dim, num_heads=8, norm_layer=nn.LayerNorm, hidden_tokens=None, act_layer=nn.GELU,
                 drop=0., k=4):
        super().__init__()
        assert inter_token % num_heads == 0, f"dim {inter_token} should be divided by num_heads {num_heads}."

        self.inter_token = inter_token
        self.num_heads = num_heads
        head_dim = inter_token // num_heads
        self.scale = None or head_dim ** -0.5
        self.k = k  # 1, 2, 4
        self.drop = nn.Dropout(drop)
        self.act = act_layer()
        self.hidden_features = hidden_tokens

        # attn1
        self.norm1 = norm_layer(dim // self.k)
        self.q1 = nn.Linear(inter_token, inter_token)
        self.kv1 = nn.Linear(inter_token, inter_token * 2)
        self.proj1 = nn.Linear(inter_token, inter_token)

        # attn2
        self.norm2 = norm_layer(dim // self.k)
        self.q2 = nn.Linear(inter_token, inter_token)
        self.kv2 = nn.Linear(inter_token, inter_token * 2)
        self.proj2 = nn.Linear(inter_token, inter_token)

        # conv block1
        self.linear1 = nn.Linear(inter_token, hidden_tokens)  # hidden_tokens = inter_token * mlp_ratio
        self.linear2 = nn.Linear(hidden_tokens, inter_token)
        # conv block2
        self.linear3 = nn.Linear(inter_token, hidden_tokens)  # hidden_tokens = inter_token * mlp_ratio
        self.linear4 = nn.Linear(hidden_tokens, inter_token)

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
        K = self.k
        B, D, T = x.shape  # B, D, T
        # 窗口切片
        x = x.reshape(B, K, D // K, T)

        # attn1
        shortcut1 = x  # B, K, D//K, T
        x = self.norm1(x.transpose(2, 3)).transpose(2, 3)
        q1 = self.q1(x).reshape(B, K, D // K, self.num_heads, T // self.num_heads).permute(0, 1, 3, 2,
                                                                                           4)  # B, K, heads, D//K, T//heads
        kv1 = self.kv1(x).reshape(B, K, D // K, 2, self.num_heads, T // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        k1, v1 = kv1[0], kv1[1]  # B, K, heads, D//K, T//heads

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)  # B, K, heads, D//K, D//K

        x = (attn1 @ v1).transpose(2, 3).reshape(B, K, D // K,
                                                 T)  # B, K, heads, D//K, T//heads -> B, K, D//K, heads, T//heads -> B, K, D//K, T
        x = self.proj1(x)
        x = shortcut1 + x  # B, K, D//K, T

        # MLP_CONV2D_1
        shortcut_mlp1 = x
        x = self.linear1(x)  # B, K, D//K, T
        x = self.act(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)
        x = shortcut_mlp1 + x

        # # 特征重组
        x = x.reshape(B, K, K, D // (K * K), T)
        x = x.permute(0, 2, 1, 3, 4).flatten(2, 3)

        # attn2
        shortcut2 = x
        x = self.norm2(x.transpose(2, 3)).transpose(2, 3)
        q2 = self.q2(x).reshape(B, K, D // K, self.num_heads, T // self.num_heads).permute(0, 1, 3, 2,
                                                                                           4)  # B, K, heads, D//K, T//heads
        kv2 = self.kv2(x).reshape(B, K, D // K, 2, self.num_heads, T // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        k2, v2 = kv2[0], kv2[1]  # B, K, heads, D//K, T//heads

        attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)  # B, K, heads, D//K, D//K

        x = (attn2 @ v2).transpose(2, 3).reshape(B, K, D // K,
                                                 T)  # B, K, heads, D//K, T//heads -> B, K, D//K, heads, T//heads -> B, K, D//K, T
        x = self.proj2(x)
        x = shortcut2 + x  # B, K, D//K, T

        # MLP_CONV2D_2
        shortcut_mlp2 = x
        x = self.linear3(x)  # B, K, D//K, T
        x = self.act(x)
        x = self.drop(x)
        x = self.linear4(x)
        x = self.drop(x)
        x = shortcut_mlp2 + x

        x = x.flatten(1, 2)

        return x


class GLRBlock(nn.Module):
    """
    Global Local Relational Block
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 inter_token=None, k=4):
        super().__init__()
        mlp_hidden_dim = int(inter_token * mlp_ratio)
        self.Global_Relational_Block = Global_Relational_Block_Win(
            inter_token, dim, num_heads=num_heads, norm_layer=norm_layer, hidden_tokens=mlp_hidden_dim,
            act_layer=act_layer, drop=drop, k=k)

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
        # B, T, D -> 32, 256, 256
        # B, T, D -> 32, 128, 384
        # B, T, D -> 32, 64, 576
        # B, T, D -> 32, 32, 864
        x = x.transpose(1, 2)  # B, D, T
        x = self.Global_Relational_Block(x)
        x = x.transpose(1, 2)
        return x


class Temporal_Merging_Block(nn.Module):
    """
    Temporal_Merging_Block
    """

    def __init__(self, kernel_size=3, stride=1, in_chans=1024, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride,
                              padding=(kernel_size // 2))
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

        # 32, 1024, 256
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class RelationEncoder(nn.Module):
    def __init__(self, in_feat_dim=1024, embed_dims=[256, 384, 576, 864],
                 num_head=8, mlp_ratio=8, norm_layer=nn.LayerNorm,
                 num_block_spatial=None, inter_tokens=[256, 128, 64, 32], k=4):
        super().__init__()

        # Stage 1
        if num_block_spatial is None:
            num_block_spatial = [1, 1, 3, 1]
        self.Temporal_Merging_Block1 = Temporal_Merging_Block(kernel_size=3, stride=1, in_chans=in_feat_dim,
                                                              embed_dim=embed_dims[0])
        self.block1 = nn.ModuleList([GLRBlock(
            dim=embed_dims[0], num_heads=num_head, mlp_ratio=mlp_ratio, norm_layer=norm_layer,
            inter_token=inter_tokens[0], k=k)
            for i in range(num_block_spatial[0])])
        self.norm1 = norm_layer(embed_dims[0])

        # Stage 2
        self.Temporal_Merging_Block2 = Temporal_Merging_Block(kernel_size=3, stride=2, in_chans=embed_dims[0],
                                                              embed_dim=embed_dims[1])
        self.block2 = nn.ModuleList([GLRBlock(
            dim=embed_dims[1], num_heads=num_head, mlp_ratio=mlp_ratio, norm_layer=norm_layer,
            inter_token=inter_tokens[1], k=k)
            for i in range(num_block_spatial[1])])
        self.norm2 = norm_layer(embed_dims[1])

        # Stage 3
        self.Temporal_Merging_Block3 = Temporal_Merging_Block(kernel_size=3, stride=2, in_chans=embed_dims[1],
                                                              embed_dim=embed_dims[2])
        self.block3 = nn.ModuleList([GLRBlock(
            dim=embed_dims[2], num_heads=num_head, mlp_ratio=mlp_ratio, norm_layer=norm_layer,
            inter_token=inter_tokens[2], k=k)
            for i in range(num_block_spatial[2])])
        self.norm3 = norm_layer(embed_dims[2])

        # Stage 4
        self.Temporal_Merging_Block4 = Temporal_Merging_Block(kernel_size=3, stride=2, in_chans=embed_dims[2],
                                                              embed_dim=embed_dims[3])
        self.block4 = nn.ModuleList([GLRBlock(
            dim=embed_dims[3], num_heads=num_head, mlp_ratio=mlp_ratio, norm_layer=norm_layer,
            inter_token=inter_tokens[3], k=k)
            for i in range(num_block_spatial[3])])
        self.norm4 = norm_layer(embed_dims[3])

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

    def freeze_init_emb(self):
        self.Temporal_Merging_Block1.requires_grad = False

    def forward(self, x):
        outs = []
        # stage 1
        # B, D, T -> 32, 1024, 32
        x = self.Temporal_Merging_Block1(x)
        # B, T, D -> 32, 32, 256
        for i, blk in enumerate(self.block1):
            x = blk(x)
        x = self.norm1(x)
        # 32, 32, 256
        x = x.permute(0, 2, 1).contiguous()
        # B, D, T -> 32, 256, 32
        outs.append(x)

        # stage 2
        # B, D, T -> 32, 256, 32
        x = self.Temporal_Merging_Block2(x)
        # B, T, D -> 32, 16, 384
        for i, blk in enumerate(self.block2):
            x = blk(x)
        x = self.norm2(x)
        # B, T, D -> 32, 16, 384
        x = x.permute(0, 2, 1).contiguous()
        # B, D, T -> 32, 384, 16
        outs.append(x)

        # stage 3
        # B, D, T -> 32, 384, 16
        x = self.Temporal_Merging_Block3(x)
        # B, T, D -> 32, 8, 576
        for i, blk in enumerate(self.block3):
            x = blk(x)
        x = self.norm3(x)
        # B, T, D -> 32, 8, 576
        x = x.permute(0, 2, 1).contiguous()
        # B, D, T -> 32, 576, 8
        outs.append(x)

        # stage 4
        # B, D, T -> 32, 576, 8
        x = self.Temporal_Merging_Block4(x)
        # B, T, D -> 32, 4, 864
        for i, blk in enumerate(self.block4):
            x = blk(x)
        x = self.norm4(x)
        # B, T, D -> 32, 4, 864
        x = x.permute(0, 2, 1).contiguous()
        # B, D, T -> 32, 864, 4
        outs.append(x)

        return outs
