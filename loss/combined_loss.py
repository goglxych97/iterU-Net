import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
import .perceptual as ploss

loss_L1 = torch.nn.L1Loss()
loss_SSIM = lambda x, y: 1 - ssim(x, y)
loss_PP = ploss.MedicalNetPerceptualSimilarity()

def gradient_difference_loss(prediction, target):
    def compute_gradient(tensor):
        dz = torch.abs(tensor[:, :, 1:, :, :] - tensor[:, :, :-1, :, :])
        dy = torch.abs(tensor[:, :, :, 1:, :] - tensor[:, :, :, :-1, :])
        dx = torch.abs(tensor[:, :, :, :, 1:] - tensor[:, :, :, :, :-1])
        return dx, dy, dz

    pred_dx, pred_dy, pred_dz = compute_gradient(prediction)
    tgt_dx, tgt_dy, tgt_dz = compute_gradient(target)

    return (
        F.l1_loss(pred_dx, tgt_dx)
        + F.l1_loss(pred_dy, tgt_dy)
        + F.l1_loss(pred_dz, tgt_dz)
    )


def compute_total_loss(outputs, targets, mask=None, device="cuda"):
    loss_PP_fn = loss_PP.to(device)
    total_l1, total_ssim, total_pp, total_gd = 0, 0, 0, 0

    for out, tgt in zip(outputs, targets):
        if mask is not None:
            out, tgt = out * mask, tgt * mask

        l1 = 0.1 * loss_L1(out, tgt)
        ssim_val = loss_SSIM(out, tgt)
        pp = loss_PP_fn(out, tgt).squeeze().mean()
        gd = 0.1 * gradient_difference_loss(out, tgt)

        total_l1 += l1
        total_ssim += ssim_val
        total_pp += pp
        total_gd += gd

    total = total_l1 + total_ssim + total_pp + total_gd

    return dict(
        total=total,
        l1=total_l1,
        ssim=total_ssim,
        pp=total_pp,
        gd=total_gd,
    )
