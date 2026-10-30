#!/usr/bin/env python3
"""sigmake.py: byte patterns for the VR fork's version-independent hooks (vr_camera.cpp's Games).

    sigmake.py func ELF ADDR             the opening of a function (>= 24 bytes), unique in the code
    sigmake.py call ELF ADDR             the instructions before a call at ADDR, ending with the call (>= 14 bytes)
    sigmake.py patch ELF ADDR LEN        code around [ADDR, ADDR+LEN), the patched bytes exact, unique; prints the
                                         offset of ADDR in the pattern

ELF is prepare.py's plaintext ELF (out/<game>/eboot.elf); ADDR is a virtual address (hex). A pattern is hex bytes with
"??" for what moves with the code: call and jump targets and rip-relative displacements (jumps inside a patched range
stay exact). Needs capstone.
"""
import re
import struct
import sys
from pathlib import Path

import capstone

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import vr_resolution as v  # noqa: E402

md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
md.detail = True


def decode(code, va, start, end, exact=(0, 0)):
    """[(address, size, [(offset, length) to mask])] of the instructions from start to end."""
    out = []
    for i in md.disasm(code[start - va:start - va + (end - start) + 16], start):
        if i.address >= end:
            break
        mask = []
        if i.disp_size and any(o.type == capstone.x86.X86_OP_MEM and o.mem.base == capstone.x86.X86_REG_RIP
                               for o in i.operands):
            mask.append((i.disp_offset, i.disp_size))
        inside = i.address < exact[1] and i.address + i.size > exact[0]
        if (i.group(capstone.CS_GRP_CALL) or (i.group(capstone.CS_GRP_JUMP) and not inside)) and i.imm_size == 4:
            mask.append((i.imm_offset, 4))
        out.append((i.address, i.size, mask))
    return out


def pattern(code, va, start, end, exact=(0, 0)):
    ins = decode(code, va, start, end, exact)
    if not ins or ins[0][0] != start or ins[-1][0] + ins[-1][1] != end:
        return None
    pat = []
    for a, size, mask in ins:
        b = list(code[a - va:a - va + size])
        for o, n in mask:
            for k in range(n):
                b[o + k] = None
        pat += b
    return pat


def regex(pat):
    return re.compile(b''.join(b'.' if b is None else re.escape(bytes([b])) for b in pat), re.S)


def matches(code, pat):
    r, pos, out = regex(pat), 0, []
    while True:
        m = r.search(code, pos)
        if not m:
            return out
        out.append(m.start())
        pos = m.start() + 1


def text(pat):
    return ' '.join('??' if b is None else f'{b:02x}' for b in pat)


def boundary(code, va, addr, back):
    """The earliest instruction boundary at least `back` bytes before addr that two decodes agree on."""
    for k in range(back, back + 12):
        one = decode(code, va, addr - k, addr)
        if not one or one[-1][0] + one[-1][1] != addr:
            continue
        two = decode(code, va, addr - k - 24, addr)
        if not two or two[-1][0] + two[-1][1] != addr:
            continue
        agreed = {a for a, _s, _m in one} & {a for a, _s, _m in two}
        if agreed:
            return min(a for a in agreed if a >= addr - k + 8) if any(a >= addr - k + 8 for a in agreed) else None
    return None


def end_after(code, va, addr, length):
    ins = decode(code, va, addr, addr + length + 16)
    return next((a + s for a, s, _m in ins if a + s >= addr + length), None)


def func(code, va, addr):
    for n in range(24, 120):
        end = end_after(code, va, addr, n)
        pat = pattern(code, va, addr, end) if end else None
        if pat and len(pat) >= 24 and len(matches(code, pat)) == 1:
            return pat, 0
    return None, 0


def call(code, va, addr):
    start0 = boundary(code, va, addr, 70)
    starts = [a for a, _s, _m in decode(code, va, start0, addr + 5) if addr - a >= 14]
    for start in reversed(starts):
        pat = pattern(code, va, start, addr + 5)
        if not pat:
            continue
        callers = [m for m in matches(code, pat)
                   if va + m + len(pat) + struct.unpack_from('<i', code, m + len(pat) - 4)[0] == _target(code, va, addr)]
        if len(matches(code, pat)) == 1 and len(callers) == 1:
            return pat, addr - start
    return None, 0


def _target(code, va, addr):
    return addr + 5 + struct.unpack_from('<i', code, addr - va + 1)[0]


def patch(code, va, addr, length):
    for before in range(8, 60, 2):
        start = boundary(code, va, addr, before)
        if start is None:
            continue
        for after in range(6, 60, 2):
            end = end_after(code, va, addr, length + after)
            pat = pattern(code, va, start, end, (addr, addr + length)) if end else None
            if pat and len(matches(code, pat)) == 1:
                return pat, addr - start
    return None, 0


def main():
    mode, path, addr = sys.argv[1], sys.argv[2], int(sys.argv[3], 16)
    elf = Path(path).read_bytes()
    va, code = v.code_segment(elf)
    if mode == 'func':
        pat, at = func(code, va, addr)
    elif mode == 'call':
        pat, at = call(code, va, addr)
    else:
        pat, at = patch(code, va, addr, int(sys.argv[4]))
    if pat is None:
        print('no unique pattern')
        return 1
    print(f'{text(pat)}\noffset {at} length {len(pat)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
