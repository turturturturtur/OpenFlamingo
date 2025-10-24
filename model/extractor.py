import torch
import torch.nn as nn
from registry import MODEL

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None, mask_location=None):
        assert (mask is None and mask_location is None) or (
                mask is not None and mask_location is not None), f'{mask} {mask_location}'
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # Mask Implementation
            # B: # batches, N: # tokens, C: embed_dim, H: # heads
            # mask: (B, N)
            # attn: (B, H, N, N)
            # mask = torch.tril(torch.ones(64, 197), diagonal=0).to(attn.device)
            if mask_location == 'pre-softmax':

                attn_permute = attn.permute((1, 0, 3, 2))
                attn_permute[:, mask != 1] = -torch.tensor(float('inf')).type(attn_permute.dtype).to(attn.device)
                attn = attn_permute.permute((1, 0, 3, 2))
                attn = attn.softmax(dim=-1)
                # attn = attn + (1 - mask).masked_fill(mask != 1, float("-Inf")).unsqueeze(1).unsqueeze(2)
                # attn = attn.softmax(dim=-1)
            elif mask_location == 'pre-softmax_':
                attn = attn + (1 - mask).masked_fill(mask != 1, float("-Inf")).unsqueeze(1).unsqueeze(2)
                attn = attn.softmax(dim=-1)
            elif mask_location == 'post-softmax':
                attn = attn.softmax(dim=-1)
                attn = attn * mask.unsqueeze(1).unsqueeze(2)
            else:
                raise ValueError("mask_location should be either 'pre-softmax' or 'post-softmax'")
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

@MODEL.register("extractor")
class Extractor(nn.Module):
    def __init__(self, dim_attn1=1024, dim_attn2=512, hidden_layer=256, num_heads=8, output_dim=128, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.attention1 = Attention(dim_attn1, num_heads, qkv_bias, attn_drop, proj_drop)
        self.attention2 = Attention(dim_attn2, num_heads, qkv_bias, attn_drop, proj_drop)
        self.gelu = nn.GELU()
        self.linear = nn.Linear(dim_attn1, dim_attn2)
        self.mlp = MLP(in_features=dim_attn2, hidden_features=hidden_layer, out_features=output_dim, drop=proj_drop)
        self.layernorm = nn.LayerNorm(dim_attn2)

    def forward(self, x, mask=None, mask_location=None):
        # 输入 x 形状: [B, N, dim_attn1]
        
        x = self.attention1(x, mask, mask_location)
        x = self.gelu(x)
        x = self.linear(x)
        # x 形状: [B, N, dim_attn2]
        
        x = self.attention2(x, mask, mask_location)
        x = self.layernorm(x)
        # x 形状: [B, N, dim_attn2]

        # --- [关键修改] 在这里进行池化操作 ---
        # 将一个批次的特征序列，通过在序列维度(dim=1)上取平均，
        # 聚合成一个批次的单一特征向量。
        x_pooled = x.mean(dim=1)
        # x_pooled 形状: [B, dim_attn2]
        
        # 现在可以将正确的二维张量喂给 MLP
        x = self.mlp(x_pooled)
        # 输出 x 形状: [B, output_dim]
        
        return x