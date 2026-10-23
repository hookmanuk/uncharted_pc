"""size_find.py BASE SIZE: u32 and float (1920, 1080) pairs (and (960, 540)) in [BASE, BASE+SIZE) of the running game."""
import sys
import numpy as np
import memscan

base, size = int(sys.argv[1], 16), int(sys.argv[2], 16)
data = memscan.read(base, size)
u = np.frombuffer(data[:len(data) // 4 * 4], dtype='<u4')
f = np.frombuffer(data[:len(data) // 4 * 4], dtype='<f4')
for w, h in ((1920, 1080), (960, 540)):
    for name, arr, a, b in (('u32', u, w, h), ('f32', f, np.float32(w), np.float32(h))):
        idx = np.nonzero((arr[:-1] == a) & (arr[1:] == b))[0]
        for i in idx[:40]:
            print(f'{name} {w}x{h} at image +{i * 4:#x}')
