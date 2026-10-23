"""imm_find.py BASE CODE_SIZE IMM [IMM...]: instructions using these immediates in the game image's code."""
import sys
import capstone
import memscan

base, size = int(sys.argv[1], 16), int(sys.argv[2], 16)
want = {int(a, 0) for a in sys.argv[3:]}
md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
md.detail = True
md.skipdata = True
for ins in md.disasm(memscan.read(base, size), base):
    if ins.id == 0:  # skipped data
        continue
    for op in ins.operands:
        if op.type == capstone.x86.X86_OP_IMM and op.imm in want:
            print(f'+{ins.address - base:#x}: {ins.mnemonic} {ins.op_str}')
            break
