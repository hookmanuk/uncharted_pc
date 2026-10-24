"""view_find.py ELF: candidates for the camera view builder (Uncharted 2's +0x3e2c50 shape) in a plaintext game ELF:
vblendps xmm, xmm, [reg + P], 7 (the position, w from a constant) with vandps of [reg + P - 0x30/-0x20/-0x10] (the
three rows) before it, and the function start (push rbp; mov rbp, rsp) above."""
import re
import struct
import sys
from pathlib import Path
import capstone

elf = Path(sys.argv[1]).read_bytes()
phoff, = struct.unpack_from('<Q', elf, 0x20)
phentsize, phnum = struct.unpack_from('<HH', elf, 0x36)
md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
md.skipdata = True
for i in range(phnum):
    p_type, p_flags, p_offset, p_vaddr, _, p_filesz = struct.unpack_from('<IIQQQQ', elf, phoff + i * phentsize)
    if p_type != 1 or not (p_flags & 1):
        continue
    code = elf[p_offset:p_offset + p_filesz]
    # VEX3 vblendps (0F3A 0C), modrm mod=10 (disp32), imm8 7
    for m in re.finditer(rb'\xc4[\x43\x63\xc3\xe3][\x00-\xff]\x0c[\x80-\xbf][\x00-\xff]{4}\x07', code, re.S):
        at = m.start()
        disp, = struct.unpack_from('<i', code, at + 5)
        window = code[max(0, at - 0x60):at + 0x60]
        base = p_vaddr + max(0, at - 0x60)
        rows = set()
        for ins in md.disasm(window, base):
            if ins.mnemonic == 'vandps' and '+ 0x' in ins.op_str:
                rows.add(int(ins.op_str.split('+ 0x')[1].rstrip(']'), 16))
        if not {disp - 0x30, disp - 0x20, disp - 0x10} <= rows:
            continue
        start = None
        for back in range(at, max(0, at - 0x400), -1):
            if code[back:back + 4] == b'\x55\x48\x89\xe5':
                start = p_vaddr + back
                break
        print(f'+{p_vaddr + at:#x}: position +{disp:#x}, rows +{disp - 0x30:#x}; function '
              f'{"+%#x" % start if start else "?"}: {code[start - p_vaddr:start - p_vaddr + 16].hex(" ") if start else ""}')
