"""xrefs.py BASE CODE_SIZE LO HI [CALL_TARGET]: RIP-relative references to [BASE+LO, BASE+HI) in the game image's code
(BASE+0 .. BASE+CODE_SIZE), and calls to BASE+CALL_TARGET; offsets relative to BASE."""
import sys
import capstone
import memscan

base, size, lo, hi = (int(a, 16) for a in sys.argv[1:5])
call = int(sys.argv[5], 16) if len(sys.argv) > 5 else None
code = memscan.read(base, size)
md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
md.detail = True
md.skipdata = True
for ins in md.disasm(code, base):
    if ins.mnemonic == 'call' and call is not None and ins.op_str == hex(base + call):
        print(f'call  +{ins.address - base:#x}')
    if '[rip' not in ins.op_str:
        continue
    for op in ins.operands:
        if op.type == capstone.x86.X86_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP:
            target = ins.address + ins.size + op.mem.disp
            if base + lo <= target < base + hi:
                print(f'ref   +{ins.address - base:#x}: {ins.mnemonic} {ins.op_str}  -> +{target - base:#x}')
