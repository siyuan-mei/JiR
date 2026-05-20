from typing import Sequence
from monai.networks.nets.swin_unetr import SwinUNETR
from ..register import MODELS


@MODELS.register(type='swinunetr')
class SwinUNETRWrapper(SwinUNETR):
    def __init__(self,
                 img_size: Sequence[int] | int,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 spatial_dims: int = 3,
                 feature_size: int = 48,
                 norm_name: tuple | str = "instance",
                 **kwargs):
        super().__init__(img_size=img_size, in_channels=in_channels, out_channels=out_channels,
                         spatial_dims=spatial_dims,
                         feature_size=feature_size, norm_name=norm_name, **kwargs)
