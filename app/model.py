"""RRDBNet x4 inference compatible with official RealESRGAN_x4plus params_ema.
Architecture based on xinntao/Real-ESRGAN and XPixelGroup/BasicSR (BSD-3-Clause).
"""
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

WEIGHTS_URL = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'

class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2*gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3*gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4*gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x + x5 * .2

class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)
    def forward(self, x):
        return x + self.rdb3(self.rdb2(self.rdb1(x))) * .2

class RRDBNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_first = nn.Conv2d(3, 64, 3, 1, 1)
        self.body = nn.Sequential(*(RRDB() for _ in range(23)))
        self.conv_body = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(.2, inplace=True)
    def forward(self, x):
        feat = self.conv_first(x)
        feat = self.conv_body(self.body(feat)) + feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))

class Upscaler:
    """Fast tiled RRDBNet x4 inference; no temporal shortcuts or denoising blur."""

    def __init__(self, weight_path, tile=0, device=None, precision='fp16', tile_pad=16):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        if precision not in ('fp16', 'fp32'):
            raise ValueError('precision phải là fp16 hoặc fp32')
        self.half = self.device.type == 'cuda' and precision == 'fp16'
        self.dtype = torch.float16 if self.half else torch.float32
        if tile == 0:
            memory_gb = (torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
                         if self.device.type == 'cuda' else 0)
            tile = 384 if memory_gb >= 20 else 256 if memory_gb >= 10 else 128
        self.tile = int(tile)
        self.tile_pad = int(tile_pad)
        if not 32 <= self.tile <= 1024:
            raise ValueError('Tile phải từ 32 đến 1024')
        if not 0 <= self.tile_pad <= 64:
            raise ValueError('tile_pad phải trong khoảng 0 đến 64')
        self.net = RRDBNet()
        try:
            ckpt = torch.load(str(weight_path), map_location='cpu', weights_only=True)
        except TypeError:
            ckpt = torch.load(str(weight_path), map_location='cpu')
        weights = ckpt.get('params_ema', ckpt.get('params', ckpt))
        self.net.load_state_dict(weights, strict=True)
        self.net.eval().to(self.device)
        if self.half:
            self.net.half()
        if self.device.type == 'cuda':
            # The same tile size is reused throughout a video; cuDNN can autotune.
            torch.backends.cudnn.benchmark = True

    @torch.inference_mode()
    def _run(self, rgb):
        # Transfer uint8 instead of float32 (4x less upload bandwidth).
        inp = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0)
        inp = inp.to(self.device, dtype=self.dtype).div_(255.)
        # Convert to uint8 on GPU before copying prediction back to CPU.
        pred = self.net(inp).clamp_(0, 1).mul_(255).add_(0.5).to(torch.uint8)
        return pred[0].permute(1, 2, 0).contiguous().cpu().numpy()

    def _enhance_with_tile(self, src, h, w, tile):
        hh, ww = src.shape[:2]
        out = np.empty((hh*4, ww*4, 3), dtype=np.uint8)
        pad = self.tile_pad
        for y in range(0, hh, tile):
            for x in range(0, ww, tile):
                y0 = max(0, y-pad); x0 = max(0, x-pad)
                y1 = min(hh, y+tile+pad); x1 = min(ww, x+tile+pad)
                pred = self._run(src[y0:y1, x0:x1])
                crop = pred[(y-y0)*4:(min(y+tile,hh)-y0)*4,
                            (x-x0)*4:(min(x+tile,ww)-x0)*4]
                out[y*4:min(y+tile,hh)*4, x*4:min(x+tile,ww)*4] = crop[:, :, ::-1]
        return out[:h*4, :w*4]

    def enhance(self, image_bgr):
        h, w = image_bgr.shape[:2]
        src = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pad_h = (4-h%4)%4; pad_w = (4-w%4)%4
        if pad_h or pad_w:
            src = np.pad(src, ((0,pad_h),(0,pad_w),(0,0)), mode='edge')
        while True:
            try:
                return self._enhance_with_tile(src, h, w, self.tile)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if 'out of memory' not in str(e).lower() or self.device.type != 'cuda' or self.tile <= 64:
                    raise
                new_tile = max(64, self.tile // 2)
                print(f'VRAM thiếu, tự giảm GPU tile {self.tile} → {new_tile}.', flush=True)
                self.tile = new_tile
                torch.cuda.empty_cache()
