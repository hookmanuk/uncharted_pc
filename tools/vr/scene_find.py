"""scene_find.py [MIN_ADDR]: the scene constants (864 bytes: [0]=far 3000 [1]=1/far [4..7]=1920 1080 1/1920 1/1080)
in the running game's memory above MIN_ADDR (hex, default 0x1000000000: guest memory), with the view matrix."""
import struct
import sys
import numpy as np
import memscan



def regions():
    """Committed readable regions, write-combined ones too (guest GPU memory)."""
    addr, mbi = 0, memscan.MBI()
    while memscan.k32.VirtualQueryEx(memscan.H, memscan.ctypes.c_void_p(addr), memscan.ctypes.byref(mbi),
                                     memscan.ctypes.sizeof(mbi)):
        if mbi.State == 0x1000 and mbi.Protect & 0xEE and not mbi.Protect & 0x100:
            yield mbi.BaseAddress, mbi.RegionSize, mbi.Protect, mbi.AllocationBase
        addr = mbi.BaseAddress + mbi.RegionSize
        if addr >= 1 << 47:
            break


if __name__ == "__main__":
    lo = int(sys.argv[1], 16) if len(sys.argv) > 1 and __name__ == "__main__" else 0x1000000000
    want = np.array([1920, 1080, 1 / 1920, 1 / 1080], dtype=np.float32)
    for base, size, prot, alloc in regions():
        if base + size <= lo:
            continue
        for off in range(0, size, 1 << 26):
            data = memscan.read(base + off, min(1 << 26, size - off))
            a = np.frombuffer(data[:len(data) // 4 * 4], dtype='<f4')
            if len(a) < 8:
                continue
            with np.errstate(all='ignore'):
                idx = np.nonzero(np.abs(a[4:len(a) - 3] - 1920) < 1e-3)[0] + 4
            for i in idx:
                if np.all(np.abs(a[i:i + 4] - want) <= np.abs(want) * 1e-4) and abs(a[i - 4] - 3000) < 1:
                    addr = base + off + (int(i) - 4) * 4
                    view = a[i - 4 + 8:i - 4 + 20]
                    print(f'{addr:#x} region {base:#x}+{size:#x} alloc {alloc:#x} prot {prot:#x} view ' +
                          ' '.join(f'{v:.3f}' for v in view))
