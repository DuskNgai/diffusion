import math

from timm.models.vision_transformer import (
    Attention,
    PatchEmbed,
    SwiGLUPacked as SwiGLU,
)
import torch
import torch.nn as nn
import torch.nn.functional as F

from coach_pl.configuration import configurable
from coach_pl.model import MODEL_REGISTRY


class TimestepEmbedder(nn.Module):

    def __init__(self, dim: int, nfreq: int = 256) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(nfreq, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.nfreq = nfreq

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
        half_dim = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim).to(device=t.device)
        args = torch.outer(t.view(-1).float(), freqs)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, : 1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t * 1000
        t_freq = self.timestep_embedding(t, self.nfreq)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):

    def __init__(self, num_classes: int, dim: int) -> None:
        super().__init__()

        self.embedding = nn.Embedding(num_classes + 1, dim)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding(labels)


def modulate(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def precompute_freqs_cis(grid_size: tuple[int, int], dim: int, theta: float = 10000.0) -> tuple[torch.Tensor]:
    """
    Precompute the trigonometric form of the base frequencies for the rotary positional encoding.

    Args:
        `grid_size` (tuple[int, int]): The grid size (H, W) of the input image.
        `dim` (int): The embedding dimension.
        `theta` (float): The base frequency.
        `grid_offset` (int): The offset of the grid.

    Returns:
        `freqs_cis` (torch.Tensor): The trigonometric form of the base frequencies.
    """

    assert dim % 4 == 0, "`dim` should be divisible by 4."
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[:(dim // 4)].float() / dim)) # [E // 4]
    y, x = torch.meshgrid(
        torch.arange(grid_size[0], device=freqs.device, dtype=torch.float32),
        torch.arange(grid_size[1], device=freqs.device, dtype=torch.float32),
        indexing="ij",
    )                                                                             # [H, W]

    freqs_y = torch.einsum("hw, e->hwe", y, freqs)                                                  # [H, W, E // 4]
    freqs_x = torch.einsum("hw, e->hwe", x, freqs)                                                  # [H, W, E // 4]
    return torch.cat([freqs_y.cos(), freqs_y.sin(), freqs_x.cos(), freqs_x.sin()], -1).unsqueeze(0) # [1, H, W, E]


def apply_rotary_emb(
    xq: torch.Tensor,        # [B, N, L, E]
    xk: torch.Tensor,        # [B, N, L, E]
    freqs_cis: torch.Tensor, # [B, L, E],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the rotary positional embedding to the query and key tensors.
    """

    xq_real_y, xq_imag_y, xq_real_x, xq_imag_x = xq.float().chunk(4, dim=-1) # [B, N, L, E // 4]
    xk_real_y, xk_imag_y, xk_real_x, xk_imag_x = xk.float().chunk(4, dim=-1) # [B, N, L, E // 4]
    cos_y, sin_y, cos_x, sin_x = freqs_cis.unsqueeze(1).chunk(4, dim=-1)     # [B, 1, L, E // 4]

    xq_out = torch.cat(
        [
            xq_real_y * cos_y - xq_imag_y * sin_y,
            xq_real_y * sin_y + xq_imag_y * cos_y,
            xq_real_x * cos_x - xq_imag_x * sin_x,
            xq_real_x * sin_x + xq_imag_x * cos_x,
        ],
        dim=-1,
    ).type_as(xq)                                  # [B, N, L, E]
    xk_out = torch.cat(
        [
            xk_real_y * cos_y - xk_imag_y * sin_y,
            xk_real_y * sin_y + xk_imag_y * cos_y,
            xk_real_x * cos_x - xk_imag_x * sin_x,
            xk_real_x * sin_x + xk_imag_x * cos_x,
        ],
        dim=-1,
    ).type_as(xk)                                  # [B, N, L, E]

    return xq_out, xk_out


class RoPEAttention(Attention):
    """
    Attention block with support for RoPE.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.RMSNorm,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L = x.shape[: 2]

        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rotary_emb(q, k, freqs_cis)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if attn_mask is not None:
                attn = attn.masked_fill(attn_mask, float("-inf"))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, L, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DiTBlock(nn.Module):

    def __init__(
        self,
        dim: int = 384,
        num_heads: int = 16,
        mlp_ratio: float = 8.0 / 3.0,
    ) -> None:
        super().__init__()

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.norm1 = nn.RMSNorm(dim)
        self.attn = RoPEAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=0.0,
            proj_drop=0.0,
            norm_layer=nn.RMSNorm,
        )
        self.norm2 = nn.RMSNorm(dim)
        self.mlp = SwiGLU(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.SiLU,
            drop=0.0,
        )

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), scale_msa, shift_msa), freqs_cis)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), scale_mlp, shift_mlp))
        return x


class FinalLayer(nn.Module):

    def __init__(self, dim: int, patch_size: int, out_dim: int) -> None:
        super().__init__()

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.norm_final = nn.RMSNorm(dim)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


@MODEL_REGISTRY.register()
class DiffusionTransformer(nn.Module):

    @configurable
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 2,
        in_chans: int = 3,
        dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 8.0 / 3.0,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        self.in_chans = in_chans

        self.x_embedder = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=dim,
            output_fmt="NHWC",
        )
        self.s_embedder = TimestepEmbedder(dim)
        self.t_embedder = TimestepEmbedder(dim)
        if num_classes > 0:
            self.c_embedder = LabelEmbedder(num_classes=num_classes, dim=dim)
        else:
            self.c_embedder = None

        self.blocks = nn.ModuleList([DiTBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(dim=dim, patch_size=patch_size, out_dim=in_chans)

        self.register_buffer(
            "freqs_cis", precompute_freqs_cis(self.x_embedder.grid_size, dim // num_heads).reshape(1, self.x_embedder.num_patches, -1)
        )

        self.initialize_weights()

    @classmethod
    def from_config(cls, cfg):
        return {
            "img_size": cfg.MODEL.IMG_SIZE,
            "in_chans": cfg.MODEL.IN_CHANS,
            "num_classes": cfg.MODEL.NUM_CLASSES,
        }

    def initialize_weights(self):
        # Initialize transformer layers:
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_init_weights)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        if self.c_embedder is not None:
            nn.init.normal_(self.c_embedder.embedding.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.s_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.s_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        c = self.in_chans
        ph, pw = self.x_embedder.patch_size

        x = x.reshape(-1, h, w, ph, pw, c)
        x = torch.einsum("bhwpqc -> bchpwq", x)
        x = x.reshape(-1, c, h * ph, h * pw)
        return x

    def forward(self, x: torch.Tensor, s: torch.Tensor, t: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            `x` (torch.Tensor): Input tensor.
            `s` (torch.Tensor): Scale tensor or previous time step tensor.
            `t` (torch.Tensor): Sigma tensor or current time step tensor.
            `c` (torch.Tensor | None, optional): Condition tensor. Defaults to None.

        Returns:
            (torch.Tensor): Output tensor.
        """

        x_embed = self.x_embedder(x) # [B, L, E]
        B, H, W, E = x_embed.shape
        x_embed = x_embed.reshape(B, H * W, E)

        s_embed = self.s_embedder(s) # [B, E]
        t_embed = self.t_embedder(t) # [B, E]
        c_embed = s_embed + t_embed

        if c is not None:
            c_embed = c_embed + self.c_embedder(c) # [B, E]

        for block in self.blocks:
            x_embed = block(x_embed, self.freqs_cis, c_embed)

        x_embed = self.final_layer(x_embed, c_embed)
        x_embed = self.unpatchify(x_embed, H, W)
        return x_embed
