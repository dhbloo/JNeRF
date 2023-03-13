import jittor as jt
from jittor import nn
from jnerf.utils.registry import LOSSES

@LOSSES.register_module()
class DistortionLoss(nn.Module):
    def __init__(self):
        pass

    def execute(self, t, w):
        """Compute iint w[i] w[j] |t[i] - t[j]| di dj."""
        # The loss incurred between all pairs of intervals.
        ut = (t[..., 1:] + t[..., :-1]) / 2
        dut = jt.abs(ut[..., :, None] - ut[..., None, :])

        loss_inter = jt.sum(w * jt.sum(w[..., None, :] * dut, dim=-1), dim=-1)

        # The loss incurred within each individual interval with itself.
        loss_intra = jt.sum(w**2 * (t[..., 1:] - t[..., :-1]), dim=-1) / 3

        loss_distortion = loss_inter + loss_intra
        return jt.mean(loss_distortion)


@LOSSES.register_module()
class TestLoss(nn.Module):
    def __init__(self):
        pass

    def execute(self, t, w):
        return jt.mean(w**2) * 10000

