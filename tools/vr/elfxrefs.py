"""elfxrefs.py ELF TARGET: RIP-relative references to TARGET (hex) in a plaintext game ELF's code."""
import sys
from pathlib import Path
import capstone
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import vr_resolution as v

elf = Path(sys.argv[1]).read_bytes()
target = int(sys.argv[2], 16)
vaddr, offset, size = next(iter(v.segments(elf)))  # the first loadable segment: the code
md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
md.detail = True
md.skipdata = True
for ins in md.disasm(elf[offset:offset + size], vaddr):
    if ins.id == 0 or '[rip' not in ins.op_str:
        continue
    for op in ins.operands:
        if op.type == capstone.x86.X86_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP:
            if ins.address + ins.size + op.mem.disp == target:
                print(f'+{ins.address:#x}: {ins.mnemonic} {ins.op_str}')
