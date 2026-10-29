"""flashcount.py [RING]: one-frame grey/white pixels in a flash ring (flashring.raw): bright (luma > 90) and grey
(max - min channel < 30) in a frame, while in both neighbouring frames the same pixel is at least 40 darker. Prints the
total per eye and the frames with the most, a number to compare rings by (one shader skipped against none)."""
import os
import struct
import sys

import numpy as np

ring = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ['TEMP'], 'dcvr', 'flashring.raw')
data = open(ring, 'rb').read()
w, h, n = struct.unpack('<III', data[:12])
frames = np.frombuffer(data[12:12 + w * h * 4 * n], np.uint8).reshape(n, h, w, 4)[..., :3].astype(np.int16)
luma = frames.mean(axis=3)
grey = (frames.max(axis=3) - frames.min(axis=3)) < 30
per_frame = []
for i in range(1, n - 1):
    hit = (luma[i] > 90) & grey[i] & (luma[i] - luma[i - 1] > 40) & (luma[i] - luma[i + 1] > 40)
    per_frame.append((int(hit[:, :w // 2].sum()), int(hit[:, w // 2:].sum()), i))
left = sum(p[0] for p in per_frame)
right = sum(p[1] for p in per_frame)
big = sum(1 for p in per_frame if p[0] + p[1] >= 200)
print(f'{n} frames: one-frame grey pixels left {left}, right {right}; flashes (200+ px): {big}')
worst = sorted(per_frame, key=lambda p: p[0] + p[1], reverse=True)[:8]
print('worst frames:', ' '.join(f'{i}:{l}/{r}' for l, r, i in worst))
