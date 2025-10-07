import torch
import torch.nn as nn
import numpy as np
from PIL import Image
# from mmcv.runner import  auto_fp16
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Grid(object):
    def __init__(self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch

    def __call__(self, img, label):
        if np.random.rand() > self.prob:
            return img, label
        h = img.size(1)
        w = img.size(2)
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d1, self.d2)
        if self.ratio == 1:
            self.l = np.random.randint(1, d)
        else:
            self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

        mask = torch.from_numpy(mask).float()
        if self.mode == 1:
            mask = 1 - mask

        mask = mask.expand_as(img)
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float()
            offset = (1 - mask) * offset
            img = img * mask + offset
        else:
            img = img * mask

        return img, label


class GridMask(nn.Module):
    def __init__(self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7):
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self.fp16_enable = False

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch  # + 1.#0.5

   # @auto_fp16()
    def forward(self, x):
        if np.random.rand() > self.prob or not self.training:
            return x
        n, c, h, w = x.size()
        x = x.view(-1, h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(2, h)
        self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

        mask = torch.from_numpy(mask).to(x.dtype).to(x.device)
        if self.mode == 1:
            mask = 1 - mask
        mask = mask.expand_as(x)
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).to(x.dtype).to(x.device)
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask

        return x.view(n, c, h, w)

class PatchGridMask(nn.Module):
    """
    GridMask in PATCH space for feature maps (x: [B, C, H, W]).
    - use_h/use_w: 是否在高/宽方向打格
    - d_min, d_max: 网格周期 d 的采样范围（单位=patch）
    - ratio: 遮挡条带宽度 l / d，0~1
    - rotate: 旋转最大角度（度数）。0 表示不旋转；>0 则在 [-rotate, +rotate] 里均匀采样
    - offset: 被遮区域用噪声填充（True），否则置零（False）
    - rescale: 期望保持缩放（除以 keep_ratio）
    - prob: 生效概率
    - same_on_batch: 同一 batch 复用同一张 mask（提升稳定性）；False 则每样本独立
    """
    def __init__(
        self,
        use_h: bool = True,
        use_w: bool = True,
        d_min: int = 4,
        d_max: int = 16,
        ratio: float = 0.5,
        rotate: float = 1.0,
        offset: bool = False,
        rescale: bool = True,
        prob: float = 0.7,
        same_on_batch: bool = False,
    ):
        super().__init__()
        assert 0.0 <= ratio < 1.0
        assert d_min >= 2 and d_max >= d_min
        self.use_h = use_h
        self.use_w = use_w
        self.d_min = d_min
        self.d_max = d_max
        self.ratio = ratio
        self.rotate = float(rotate)
        self.offset = offset
        self.rescale = rescale
        self.st_prob = float(prob)
        self.prob = float(prob)
        self.same_on_batch = same_on_batch
        self.register_buffer("_tmp", torch.tensor(0.0), persistent=False)  # 仅用于拿 device/dtype

    @torch.no_grad()
    def _make_one_mask(self, H: int, W: int, device, dtype):
        # 采样 d（周期）和条带宽度 l（单位=patch）
        d_hi = min(self.d_max, max(H, W))
        d = torch.randint(self.d_min, max(d_hi, self.d_min + 1), (1,), device=device).item()
        l = max(1, min(d - 1, int(self.ratio * d + 0.5)))

        # 起始偏移
        st_h = torch.randint(0, d, (1,), device=device).item()
        st_w = torch.randint(0, d, (1,), device=device).item()

        # 构造基础 0/1 网格（1=keep, 0=drop）
        mask = torch.ones((H, W), device=device, dtype=dtype)

        # 行/列坐标
        rows = torch.arange(H, device=device)
        cols = torch.arange(W, device=device)

        if self.use_h:
            row_flag = ((rows - st_h) % d) < l  # True 的行需要被置 0
            mask[row_flag, :] = 0
        if self.use_w:
            col_flag = ((cols - st_w) % d) < l
            mask[:, col_flag] = 0

        # 旋转（可选，小角度即可；用 grid_sample）
        if self.rotate > 0:
            angle_deg = (torch.rand(1, device=device) * 2 * self.rotate - self.rotate).item()
            angle = math.radians(angle_deg)
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            theta = torch.tensor([[cos_a, -sin_a, 0.0],
                                  [sin_a,  cos_a, 0.0]], device=device, dtype=torch.float32)
            # grid_sample 需要 4D 输入
            m = mask.unsqueeze(0).unsqueeze(0).to(torch.float32)
            grid = F.affine_grid(theta.unsqueeze(0), size=m.shape, align_corners=False)
            m = F.grid_sample(m, grid, mode="nearest", padding_mode="zeros", align_corners=False)
            mask = m.squeeze(0).squeeze(0).to(dtype)

        return mask  # [H, W], {0,1}

    def set_prob(self, epoch: int, max_epoch: int):
        # 线性爬坡
        self.prob = float(self.st_prob) * float(epoch) / max(1, int(max_epoch))

    def forward(self, x: torch.Tensor):
        if (not self.training) or (torch.rand((), device=x.device) > self.prob):
            return x

        B, C, H, W = x.shape
        dtype = x.dtype
        device = x.device

        if self.same_on_batch:
            mask = self._make_one_mask(H, W, device, dtype).expand(B, 1, H, W)
        else:
            masks = [self._make_one_mask(H, W, device, dtype) for _ in range(B)]
            mask = torch.stack(masks, dim=0).unsqueeze(1)  # [B,1,H,W]

        if self.offset:
            # 被遮区域用噪声填充；噪声尺度跟特征尺度一致会更稳
            noise = torch.randn_like(x) * (x.detach().std(dim=(2,3), keepdim=True) + 1e-6)
            y = x * mask + noise * (1 - mask)
        else:
            y = x * mask

        if self.rescale:
            keep_ratio = mask.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            y = y / keep_ratio

        return y