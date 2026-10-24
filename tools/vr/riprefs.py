"""riprefs.py ELF TARGET [TARGET...]: code that refers to TARGET (hex image offset) RIP-relatively, fast: every 4-byte
position whose value as a displacement from the next byte lands on a target (lea/mov/call/jmp with no immediate after
the displacement), then each hit disassembled to confirm."""
import struct
import sys
from pathlib import Path
import capstone
import numpy as np

elf = Path(sys.argv[1]).read_bytes()
targets = [int(a, 16) for a in sys.argv[2:]]
phoff, = struct.unpack_from('<Q', elf, 0x20)
phentsize, phnum = struct.unpack_from('<HH', elf, 0x36)
md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
for i in range(phnum):
    p_type, p_flags, p_offset, p_vaddr, _, p_filesz = struct.unpack_from('<IIQQQQ', elf, phoff + i * phentsize)
    if p_type != 1 or not (p_flags & 1):
        continue
    code = np.frombuffer(elf[p_offset:p_offset + p_filesz], dtype=np.uint8)
    n = len(code) - 4
    disp = (code[:n].astype(np.int64) | (code[1:n + 1].astype(np.int64) << 8) |
            (code[2:n + 2].astype(np.int64) << 16) | (code[3:n + 3].astype(np.int64) << 24))
    disp = np.where(disp >= 1 << 31, disp - (1 << 32), disp)
    end = np.arange(n, dtype=np.int64) + p_vaddr + 4
    for target in targets:
        for at in np.nonzero(end + disp == target)[0]:
            a = p_vaddr + int(at)
            # the instruction starts 2-4 bytes before the displacement
            for back in (3, 2, 4, 1):
                ins = next(md.disasm(elf[p_offset + int(at) - back:p_offset + int(at) + 8], a - back), None)
                if ins and ins.address + ins.size == a + 4 and 'rip' in ins.op_str or (ins and ins.mnemonic in (
                        'call', 'jmp') and ins.address + ins.size == a + 4):
                    print(f'{target:#x}: +{ins.address:#x}: {ins.mnemonic} {ins.op_str}')
                    break
