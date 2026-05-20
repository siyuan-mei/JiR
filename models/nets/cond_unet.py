from typing import Sequence, Tuple, Union, List, Optional
import torch
import torch.nn as nn
from models.register import MODELS
import math

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def _as_list(x, n: int):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x] * n

def _ntuple(v, spatial_dims: int) -> Tuple[int, ...]:
    if isinstance(v, int):
        return (v,) * spatial_dims
    if isinstance(v, (list, tuple)) and len(v) == spatial_dims:
        return tuple(int(x) for x in v)
    raise ValueError(f"Expected int or sequence(len={spatial_dims}), got {v}")

def _conv(spatial_dims: int):
    return nn.Conv3d if spatial_dims == 3 else nn.Conv2d

def _convT(spatial_dims: int):
    return nn.ConvTranspose3d if spatial_dims == 3 else nn.ConvTranspose2d

def _norm(spatial_dims: int):
    return nn.InstanceNorm3d if spatial_dims == 3 else nn.InstanceNorm2d


class FiLM(nn.Module):
    """cond -> (scale, shift) for C channels."""
    def __init__(self, emb_ch: int, ch: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_ch, 2 * ch),
        )
        # start near-identity
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(cond).chunk(2, dim=1)  # (B, C)
        while scale.ndim < h.ndim:
            scale = scale.unsqueeze(-1)
            shift = shift.unsqueeze(-1)
        return h * (1 + scale) + shift


class CondResBlock(nn.Module):
    """
    Clean residual block:
      conv(stride) -> norm -> act
      conv(1)      -> norm -> FiLM(cond) -> act
      + skip (1x1 if needed)
    """
    def __init__(
        self,
        spatial_dims: int,
        in_ch: int,
        out_ch: int,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        emb_ch: int,
        conv_bias: bool = True,
        norm_op: Optional[type[nn.Module]] = None,
        norm_kwargs: Optional[dict] = None,
        act: Optional[nn.Module] = None,
        use_film=False,
    ):
        super().__init__()
        Conv = _conv(spatial_dims)
        norm_op = norm_op or _norm(spatial_dims)
        norm_kwargs = norm_kwargs or {"eps": 1e-5, "affine": True}
        act = act or nn.LeakyReLU(inplace=True)

        pad = tuple(k // 2 for k in kernel_size)

        self.conv1 = Conv(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=pad, bias=conv_bias)
        self.norm1 = norm_op(out_ch, **norm_kwargs)
        self.act1 = act

        self.conv2 = Conv(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=pad, bias=conv_bias)
        self.norm2 = norm_op(out_ch, **norm_kwargs)

        self.use_film = use_film
        if use_film:
            self.film = FiLM(emb_ch, out_ch)
        self.act2 = act

        self.skip = None
        if in_ch != out_ch or stride != 1:
            self.skip = Conv(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=conv_bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        identity = x if self.skip is None else self.skip(x)

        h = self.act1(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        if self.use_film:
            h = self.film(h, cond)
        h = self.act2(h)

        return h + identity


# -------------------------
# Decoder conv block (no residual, count = n_conv_per_stage_decoder)
# -------------------------
class CondConvBlock(nn.Module):
    """
    conv -> norm -> FiLM(cond) -> act
    """
    def __init__(
        self,
        spatial_dims: int,
        in_ch: int,
        out_ch: int,
        kernel_size: Tuple[int, ...],
        emb_ch: int,
        conv_bias: bool = True,
        norm_op: Optional[type[nn.Module]] = None,
        norm_kwargs: Optional[dict] = None,
        act: Optional[nn.Module] = None,
        use_film=False,
    ):
        super().__init__()
        Conv = _conv(spatial_dims)
        norm_op = norm_op or _norm(spatial_dims)
        norm_kwargs = norm_kwargs or {"eps": 1e-5, "affine": True}
        act = act or nn.LeakyReLU(inplace=True)

        pad = tuple(k // 2 for k in kernel_size)
        self.conv = Conv(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=pad, bias=conv_bias)
        self.norm = norm_op(out_ch, **norm_kwargs)
        self.use_film = use_film
        if use_film:
            self.film = FiLM(emb_ch, out_ch)
        self.act = act

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.conv(x))
        if self.use_film:
            x = self.film(x, cond)
        return self.act(x)


@MODELS.register(type='CondUNet')
class CondUNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        n_stages: int = 5,
        features_per_stage: Union[int, Sequence[int]] = (32, 64, 128, 256, 512),
        kernel_sizes: Union[int, Sequence[Union[int, Sequence[int]]]] = 3,
        strides: Union[int, Sequence[Union[int, Sequence[int]]]] = (1, 2, 2, 2, 2),
        n_blocks_per_stage: Union[int, Sequence[int]] = (1, 3, 4, 6, 6),
        n_conv_per_stage_decoder: Union[int, Sequence[int]] = (1, 1, 1, 1),
        num_classes: int = 1,

        # conditioning + io consistency
        use_img_cond: bool = False,
        num_label_classes: int = 2,
        use_t_emb: bool = True,
        use_y_emb: bool = True,
        emb_channels: int = 256,

        # clean defaults
        spatial_dims: int = 3,
        conv_bias: bool = True,
        norm_op: Optional[type[nn.Module]] = None,
        norm_kwargs: Optional[dict] = None,
        act: Optional[nn.Module] = None,
    ):
        super().__init__()
        assert spatial_dims in (2, 3)
        self.spatial_dims = spatial_dims
        self.use_img_cond = use_img_cond
        self.use_t_emb = use_t_emb
        self.use_y_emb = use_y_emb
        use_film = use_t_emb or use_y_emb

        # normalize configs
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        else:
            features_per_stage = list(features_per_stage)
        assert len(features_per_stage) == n_stages

        kernel_sizes = _as_list(kernel_sizes, n_stages)

        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        else:
            n_conv_per_stage_decoder = list(n_conv_per_stage_decoder)
        assert len(n_conv_per_stage_decoder) == (n_stages - 1)

        # per-stage tuples
        ks = [_ntuple(k, spatial_dims) for k in kernel_sizes]

        Conv = _conv(spatial_dims)
        ConvT = _convT(spatial_dims)
        norm_op = norm_op or _norm(spatial_dims)
        norm_kwargs = norm_kwargs or {"eps": 1e-5, "affine": True}
        act = act or nn.LeakyReLU(inplace=True)

        # embeddings (match JiT idea: cond = t_emb + y_emb)
        if self.use_t_emb:
            self.t_embedder = TimestepEmbedder(emb_channels)
        if self.use_y_emb:
            self.y_embedder = LabelEmbedder(num_label_classes, emb_channels)

        # stem
        stem_in = input_channels + 1 if use_img_cond else input_channels
        self.stem = Conv(stem_in, features_per_stage[0], kernel_size=ks[0], stride=1,
                         padding=tuple(k // 2 for k in ks[0]), bias=conv_bias)

        # encoder stages
        self.enc = nn.ModuleList()
        in_ch = features_per_stage[0]
        for i in range(n_stages):
            out_ch = features_per_stage[i]
            blocks = nn.ModuleList()

            # first block of stage i uses stage stride (downsample), except stage0 usually stride=1 per your defaults
            first_stride = strides[i]
            blocks.append(
                CondResBlock(spatial_dims, in_ch, out_ch, ks[i], first_stride, emb_channels,
                             conv_bias=conv_bias, norm_op=norm_op, norm_kwargs=norm_kwargs, act=act, use_film=use_film)
            )
            for _ in range(n_blocks_per_stage[i] - 1):
                blocks.append(
                    CondResBlock(spatial_dims, out_ch, out_ch, ks[i], 1, emb_channels,
                                 conv_bias=conv_bias, norm_op=norm_op, norm_kwargs=norm_kwargs, act=act, use_film=use_film)
                )

            self.enc.append(blocks)
            in_ch = out_ch

        # decoder: for i = n_stages-1 -> 1
        self.upconvs = nn.ModuleList()
        self.dec = nn.ModuleList()

        for i in range(n_stages - 1, 0, -1):
            up_stride = strides[i]  # mirror encoder stride
            self.upconvs.append(
                ConvT(features_per_stage[i], features_per_stage[i - 1], kernel_size=up_stride,
                      stride=up_stride, bias=conv_bias)
            )

            # after concat: (up + skip)
            dec_blocks = nn.ModuleList()
            in_dec = features_per_stage[i - 1] + features_per_stage[i - 1]
            out_dec = features_per_stage[i - 1]

            for j in range(n_conv_per_stage_decoder[i - 1]):
                dec_blocks.append(
                    CondConvBlock(spatial_dims,
                                  in_ch=in_dec if j == 0 else out_dec,
                                  out_ch=out_dec,
                                  kernel_size=ks[i - 1],
                                  emb_ch=emb_channels,
                                  conv_bias=conv_bias,
                                  norm_op=norm_op,
                                  norm_kwargs=norm_kwargs,
                                  act=act)
                )
            self.dec.append(dec_blocks)

        # head
        self.head = Conv(features_per_stage[0], num_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def _get_cond(self, t, y) -> torch.Tensor:
        cond = 0
        if self.use_t_emb:
            if t is None:
                raise ValueError("t is required when use_t_emb=True")
            cond = cond + self.t_embedder(t)
        if self.use_y_emb:
            if y is None:
                raise ValueError("y is required when use_y_emb=True")
            cond = cond + self.y_embedder(y)
        return cond

    def forward(self, x: torch.Tensor, img_cond: Optional[torch.Tensor], t: Optional[torch.Tensor], y: Optional[torch.Tensor]):
        # keep consistent with JiT
        if self.use_img_cond and img_cond is not None:
            x = torch.cat([x, img_cond], dim=1)

        cond = self._get_cond(t, y)

        # stem
        x = self.stem(x)

        # encoder
        skips: List[torch.Tensor] = []
        h = x
        for stage_blocks in self.enc:
            for blk in stage_blocks:
                h = blk(h, cond)
            skips.append(h)

        # decoder (reverse, skip last because it's bottleneck)
        # up index 0 corresponds to stage n-1 -> n-2
        for up_i, dec_blocks in enumerate(self.dec):
            # current stage index in reverse: i = n_stages-1-up_i
            skip = skips[-2 - up_i]
            h = self.upconvs[up_i](h)
            h = torch.cat([h, skip], dim=1)
            for blk in dec_blocks:
                h = blk(h, cond)

        return self.head(h)
