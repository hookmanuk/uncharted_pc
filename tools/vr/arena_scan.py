"""arena_scan.py: the running game's HeapArena objects (top, end, memory, size << 2 | flags at +0, +8, +0x10, +0x18)
in its committed memory below 0x2000000000 (regions up to 512 MB): each arena's memory, size, use and high water (+0x20)."""
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import memscan  # noqa: E402

found = []
for base, size, *_ in memscan.regions():
    if base >= 0x2000000000 or size > (512 << 20):  # the guest's direct memory mappings: the objects are elsewhere
        continue
    data = memscan.read(base, size)
    q = np.frombuffer(data[:len(data) // 8 * 8], dtype='<u8')
    if len(q) < 5:
        continue
    top, end, memory, sized = q[:-4], q[1:-3], q[2:-2], q[3:-1]
    total = sized >> np.uint64(2)
    hit = (memory >= 0x1000000000) & (memory < 0x2000000000) & (total >= (1 << 20)) & (end == memory + total) & \
          (top >= memory) & (top <= end)
    for i in np.nonzero(hit)[0]:
        found.append((base + int(i) * 8, int(memory[i]), int(total[i]), int(top[i] - memory[i]), int(q[i + 4])))
for at, memory, total, used, high in sorted(found, key=lambda f: f[1]):
    print(f'arena @{at:#x}: memory {memory:#x} size {total / 2**20:9.1f} MB used {used / 2**20:9.1f} MB '
          f'high water {high / 2**20:9.1f} MB')
