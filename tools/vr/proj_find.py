"""proj_find.py P00 P11 [TOL]: 4x4 projection matrices (P00 at [0], P11 at [5], zeros at [1..4]) in the running game."""
import sys
import numpy as np
import memscan
from scene_find import regions

p00, p11 = float(sys.argv[1]), float(sys.argv[2])
tol = float(sys.argv[3]) if len(sys.argv) > 3 else 2e-3
hits = 0
for base, size, prot, alloc in regions():
    for off in range(0, size, 1 << 26):
        data = memscan.read(base + off, min(1 << 26, size - off))
        a = np.frombuffer(data[:len(data) // 4 * 4], dtype='<f4')
        if len(a) < 16:
            continue
        with np.errstate(all='ignore'):
            idx = np.nonzero(np.abs(a[:-16] - p00) < tol)[0]
        for i in idx:
            m = a[i:i + 16]
            if abs(m[5] - p11) < tol and not np.any(m[1:5]):
                print(f'{base + off + int(i) * 4:#x} alloc {alloc:#x} prot {prot:#x}: ' + ' '.join(f'{v:.4f}' for v in m))
                hits += 1
                if hits > 60:
                    sys.exit()
