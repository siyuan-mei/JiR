import torch
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    DivisiblePadd,
    EnsureTyped,
    RandFlipd,
    Resized,
    ScaleIntensityd,
)

def build_train_pipeline(
    resize_size=None, crop_size=(112, 112, 112)  #, patch_based, patch_size=None, num_samples=1
):
    keys = ["input", "target",]
    mode = ["trilinear", "trilinear"]
    dtype = [torch.float32, torch.float32]
    # if patch_based:
    #     transforms = [
    #         CenterSpatialCropd(keys=keys, roi_size=crop_size),
    #         RandSpatialCropSamplesd(
    #             keys=keys,
    #             roi_size=patch_size,
    #             num_samples=num_samples,
    #         ),
    #         ScaleIntensityd(keys=keys, minv=0.0, maxv=1.0),
    #         RandFlipd(keys=keys, spatial_axis=2, prob=0.5),
    #         SpatialPadd(keys=keys, spatial_size=patch_size),
    #         EnsureTyped(keys=keys, data_type="tensor", dtype=dtype),
    #     ]
    # else:
    if resize_size is not None:
        transforms = [
            CenterSpatialCropd(keys=keys, roi_size=crop_size),
            DivisiblePadd(keys, k=32),
            Resized(keys=keys, spatial_size=resize_size, mode=mode),
            ScaleIntensityd(keys=keys, minv=0.0, maxv=1.0),
            # ScaleIntensityRangePercentilesd(keys=keys, lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
            RandFlipd(keys=keys, spatial_axis=2, prob=0.5),
            EnsureTyped(keys=keys, data_type="tensor", dtype=dtype),
        ]
    else:
        transforms = [
            CenterSpatialCropd(keys=keys, roi_size=crop_size),
            ScaleIntensityd(keys=keys, minv=0.0, maxv=1.0),
            # ScaleIntensityRangePercentilesd(keys=keys, lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
            RandFlipd(keys=keys, spatial_axis=2, prob=0.5),
            DivisiblePadd(keys, k=32),
            EnsureTyped(keys=keys, data_type="tensor", dtype=dtype),
        ]
    return Compose(transforms)


def build_val_pipeline(
    resize_size=None,
    crop_size=(112, 112, 112),
):
    keys = ["input", "target",]
    mode = ["trilinear", "trilinear"]
    dtype = [torch.float32, torch.float32]
    if resize_size is not None:
        transforms = [
            CenterSpatialCropd(keys=keys, roi_size=crop_size),
            DivisiblePadd(keys, k=32),
            Resized(keys=keys, spatial_size=resize_size, mode=mode),
            # ScaleIntensityRangePercentilesd(keys=keys, lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
            ScaleIntensityd(keys=keys, minv=0.0, maxv=1.0),
            EnsureTyped(keys=keys, data_type="tensor", dtype=dtype),
        ]
    else:
        transforms = [
            CenterSpatialCropd(keys=keys, roi_size=crop_size),
            # ScaleIntensityRangePercentilesd(keys=keys, lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
            ScaleIntensityd(keys=keys, minv=0.0, maxv=1.0),
            DivisiblePadd(keys, k=32),
            EnsureTyped(keys=keys, data_type="tensor", dtype=dtype)
        ]
    return Compose(transforms)



# def build_train_ct_pipeline(patch_size, num_samples=1):
#     keys = ["input", "target", "mask"]
#     dtype = [torch.float32, torch.float32, torch.uint8]
#     transforms = [
#         CropForegroundd(keys=keys, margin=0, source_key="mask", allow_smaller=True),
#         # RandCropByPosNegLabeld(
#         #     keys=keys,
#         #     label_key="mask",
#         #     spatial_size=crop_size,
#         #     pos=1.0,
#         #     neg=0.0,
#         #     num_samples=num_samples,
#         #     allow_smaller=True,
#         # ),
#         RandSpatialCropSamplesd(
#             keys=keys,
#             roi_size=patch_size,
#             num_samples=num_samples,
#         ),
#         RandFlipd(keys=keys, spatial_axis=2, prob=0.5),
#         SpatialPadd(keys=keys, spatial_size=patch_size),
#         EnsureTyped(keys=keys, data_type="tensor", dtype=dtype),
#     ]
#     return Compose(transforms)
#
#
# def build_val_ct_pipeline():
#     keys = ["input", "target", "mask"]
#     dtype = [torch.float32, torch.float32, torch.uint8]
#     return Compose(
#         [CropForegroundd(keys=keys, margin=0, source_key="mask", allow_smaller=True),
#          EnsureTyped(keys=keys, data_type="tensor", dtype=dtype),
#         ]
#     )



