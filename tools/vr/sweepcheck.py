"""sweepcheck.py [FRAMES] [YAW_AMP_DEG] [FREQ_HZ]: judder check on the OpenXR Simulator during a running VR session.

Turns the simulator's head in a smooth yaw sweep, records FRAMES consecutive composited frames (burst capture: each
with the head pose of that frame and the pose the app submitted with its projection layer), then per frame:
  head yaw, submitted yaw, their lag; the left eye image's horizontal shift from the previous frame (phase correlation
  on its middle band) against the shift the submitted yaw change predicts. A frame whose image moved other than its
  pose says (or a pose that stood still or went back while the head moved on) is judder: the compositor places the
  image by a pose it was not rendered with.
"""
import json
import math
import os
import sys
import time

import numpy as np
from PIL import Image

d = os.path.join(os.environ['LOCALAPPDATA'], 'OpenXR-Simulator')
frames = int(sys.argv[1]) if len(sys.argv) > 1 else 90
amp = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
freq = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25


def command(name, body):
    tmp = os.path.join(d, name + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(body, f)
    os.replace(tmp, os.path.join(d, name))


def yaw_of(q):
    # OpenXR: y up, -z forward; yaw about y.
    x, y, z, w = q['qx'], q['qy'], q['qz'], q['qw']
    return math.degrees(math.atan2(2 * (w * y + x * z), 1 - 2 * (y * y + x * x)))


command('pose_sweep_command.json', {'enabled': True, 'yaw_amp_deg': amp, 'pitch_amp_deg': 0.0, 'roll_amp_deg': 0.0,
                                    'freq_hz': freq})
time.sleep(2.0)
done = os.path.join(d, 'burst_done.json')
if os.path.exists(done):
    os.remove(done)
command('burst_command.json', {'frames': frames})
for _ in range(600):
    time.sleep(0.1)
    if os.path.exists(done):
        break
command('pose_sweep_command.json', {'enabled': False})
if not os.path.exists(done):
    sys.exit('no burst (is a headset session running?)')
time.sleep(0.5)
manifest = json.load(open(os.path.join(d, 'burst_manifest.json')))
rows = manifest['frames']
fov_tan = None
prev = None
results = []
for r in rows:
    img = np.asarray(Image.open(os.path.join(d, r['file'])).convert('L'), dtype=np.float32)
    h, w = img.shape
    eye = img[h // 3:2 * h // 3, :w // 2]  # the left eye's middle band
    eye = eye - eye.mean()
    shift = None
    if prev is not None:
        f = np.fft.fft2(eye) * np.conj(np.fft.fft2(prev))
        corr = np.fft.ifft2(f / (np.abs(f) + 1e-6)).real
        _, dx = np.unravel_index(np.argmax(corr), corr.shape)
        shift = dx if dx <= eye.shape[1] // 2 else dx - eye.shape[1]
    prev = eye
    head = math.degrees(r['head']['yaw']) if abs(r['head']['yaw']) < 7 else r['head']['yaw']
    sub = yaw_of(r['submitted'])
    results.append((r['i'], r['t_ms'], head, sub, shift, w // 2))
# Pixels per degree of the left eye at its middle: from the shifts against the submitted yaw changes (least squares).
pairs = [(res[3] - results[n][3], res[4]) for n, res in enumerate(results[1:]) if res[4] is not None]
num = sum(a * b for a, b in pairs)
den = sum(a * a for a, _ in pairs) or 1
ppd = num / den
print(f'frames {len(results)}, eye width {results[0][5]} px, fitted {ppd:.2f} px per degree of submitted yaw')
bad = 0
for n, (i, t, head, sub, shift, _) in enumerate(results):
    if n == 0:
        print(f'{i:3d} {t:9.1f} ms head {head:7.2f} sub {sub:7.2f} lag {head - sub:6.2f}')
        continue
    dsub = sub - results[n - 1][3]
    dhead = head - results[n - 1][2]
    predicted = dsub * ppd
    err = shift - predicted
    flag = ''
    if abs(err) > max(2.0, abs(predicted) * 0.25):
        flag += ' IMAGE!=POSE'
    if abs(dhead) > 0.05 and (dsub == 0 or dsub * dhead < 0):
        flag += ' POSE-STALL/BACK'
    bad += bool(flag)
    print(f'{i:3d} {t:9.1f} ms head {head:7.2f} sub {sub:7.2f} lag {head - sub:6.2f} dhead {dhead:6.2f} dsub '
          f'{dsub:6.2f} shift {shift:5d} px predicted {predicted:7.1f}{flag}')
print(f'frames flagged: {bad} of {len(results) - 1}')
