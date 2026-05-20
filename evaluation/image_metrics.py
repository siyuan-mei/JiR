import torch
from monai.metrics.regression import MAEMetric, MSEMetric, PSNRMetric, SSIMMetric, MultiScaleSSIMMetric
from math import exp
import torch.nn.functional as F

mae_metric = MAEMetric()
mse_metric = MSEMetric()
psnr_metric = PSNRMetric(max_val=1.0)
ssim_metric = SSIMMetric(spatial_dims=3)
ms_ssim_metric = MultiScaleSSIMMetric(spatial_dims=3, weights=[0.3, 0.5, 0.2])


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(
        window_size, window_size, window_size
    ).float().unsqueeze(0).unsqueeze(0)
    window = _3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous()
    return window

def _ssim_3D(img1, img2, window, window_size=11, channel=1, size_average=True):
    mu1 = F.conv3d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12   = F.conv3d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(dim=(1,2,3,4))  # [B]


def custom_ssim3D(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel).to(device=img1.device, dtype=img1.dtype)
    return _ssim_3D(img1, img2, window, window_size=window_size, channel=channel, size_average=size_average)


@torch.no_grad()
def evaluate(y_pred, y):
    return {
        "mae": mae_metric(y_pred, y).item(),
        "mse": mse_metric(y_pred, y).item(),
        "psnr": psnr_metric(y_pred, y).item(),
        "ssim": ssim_metric(y_pred, y).item(),
        "ms_ssim": ms_ssim_metric(y_pred, y).item(),
        "ssim_custom": custom_ssim3D(y_pred, y).item(),
    }

