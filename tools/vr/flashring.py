"""flashring.py [RING] [OUT]: reads the VR flash ring (F10 on, F9 after a flash: %TEMP%/dcvr/flashring.raw) and finds
one-frame flashes: 16 px blocks that change in a frame while the frames either side agree with each other. Prints the
frames that have them, worst first, and writes OUT (default flashring_frames/ beside the ring) with a sheet per flagged frame
(the frame before, the flash with its blocks outlined, the frame after; both eyes) and every frame as a PNG."""
import os
import struct
import sys

import numpy as np
from PIL import Image, ImageDraw

ring = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ['TEMP'], 'dcvr', 'flashring.raw')
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(ring), 'flashring_frames')
os.makedirs(out, exist_ok=True)
data = open(ring, 'rb').read()
w, h, n = struct.unpack('<III', data[:12])
size = w * h * 4
frames = np.frombuffer(data[12:12 + size * n], np.uint8).reshape(n, h, w, 4)[..., :3]
times = np.frombuffer(data[12 + size * n:12 + size * n + 8 * n], np.uint64)
print(f'{n} frames of {w}x{h} (both eyes), {(int(times[-1]) - int(times[0])) / 1e6:.2f} s' if n > 1 else f'{n} frames')

B = 16
luma = frames.astype(np.float32).mean(axis=3)
gy, gx = h // B, w // B
blocks = luma[:, :gy * B, :gx * B].reshape(n, gy, B, gx, B).mean(axis=(2, 4))
flagged = []
for i in range(1, n - 1):
    a, b, c = blocks[i - 1], blocks[i], blocks[i + 1]
    jump = np.minimum(np.abs(b - a), np.abs(b - c))  # the frame differs from both neighbours
    settle = np.abs(c - a)                            # while they agree
    hit = (jump > 18) & (settle < jump * 0.35)
    if hit.sum() >= 3:
        flagged.append((int(hit.sum()), i, hit))
flagged.sort(reverse=True)
for count, i, _ in flagged[:20]:
    print(f'frame {i:4d}: {count} blocks flash ({(int(times[i]) - int(times[0])) / 1e3:.0f} ms in)')
if not flagged:
    print('no one-frame flashes found')
for count, i, hit in flagged[:12]:
    sheet = Image.new('RGB', (w, h * 3))
    for k, j in enumerate((i - 1, i, i + 1)):
        sheet.paste(Image.fromarray(frames[j]), (0, h * k))
    draw = ImageDraw.Draw(sheet)
    for y, x in zip(*np.nonzero(hit)):
        draw.rectangle([x * B, h + y * B, x * B + B - 1, h + y * B + B - 1], outline=(255, 0, 255))
    sheet.save(os.path.join(out, f'flash_{i:04d}.png'))
for i in range(n):
    Image.fromarray(frames[i]).save(os.path.join(out, f'frame_{i:04d}.png'))
print('written to', out)
