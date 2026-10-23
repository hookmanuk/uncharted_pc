"""res_patch.py OUT W H [FIELDS] (running game, U1): a BBPATCH2 file setting the display config initialisers' sizes
(image +0x3c4d60.. +0x3c5400: mov dword [reg+off], 1920 / 1080) to W x H, their 16:9 aspect floats to W / H.
FIELDS: comma-separated struct offsets to change (hex; default 8,18,24,38)."""
import struct
import sys
import capstone
import memscan
from scene_find import regions

out, w, h = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
fields = {int(f, 16) for f in (sys.argv[4] if len(sys.argv) > 4 else '8,18,24,38').split(',')}
image = next(alloc for base, size, prot, alloc in regions() if prot == 0x20 and base == alloc and size == 0x95e000)
lo, hi = 0x3c4d00, 0x3c5500
md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
md.detail = True
patches = []
for ins in md.disasm(memscan.read(image + lo, hi - lo), image + lo):
    if ins.mnemonic != 'mov' or len(ins.operands) != 2 or ins.operands[1].type != capstone.x86.X86_OP_IMM:
        continue
    mem, imm = ins.operands[0], ins.operands[1].imm
    if mem.type != capstone.x86.X86_OP_MEM or mem.size != 4:
        continue
    disp = mem.mem.disp
    value = None
    if imm == 0x780 and disp in fields:
        value = w
    elif imm == 0x438 and disp - 4 in fields:
        value = h
    elif imm == 0x3fe38e39 and disp - 8 in fields:  # 16/9 after a size pair
        value = struct.unpack('<I', struct.pack('<f', w / h))[0]
    if value is not None:
        patches.append((ins.address - image + ins.size - 4, struct.pack('<I', value)))
        print(f'+{ins.address - image:#x}: {ins.mnemonic} {ins.op_str} -> {value:#x}')
with open(out, 'wb') as f:
    f.write(b'BBPATCH2' + struct.pack('<QQ', 0x400000, len(patches)))
    for offset, data in patches:
        f.write(struct.pack('<QQ', offset, len(data)) + data)
print(len(patches), 'patches')
