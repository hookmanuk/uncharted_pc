"""res_sites.py ELF LO HI: the display configuration initialisers' size stores in [LO, HI) of a plaintext game ELF, as
scripts/vr_resolution.py's site tables (mov dword [rdi|rax + field], imm32)."""
import struct
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import vr_resolution as v

elf = Path(sys.argv[1]).read_bytes()
lo, hi = int(sys.argv[2], 16), int(sys.argv[3], 16)
at = v.file_offset(elf, lo, hi - lo)
code = elf[at:at + hi - lo]
sizes, scaled = [], []
for i in range(len(code) - 7):
    if code[i] != 0xc7 or code[i + 1] not in (0x40, 0x47):
        continue
    reg, field = code[i + 1], code[i + 2]
    imm, = struct.unpack_from('<I', code, i + 3)
    a = lo + i
    if (imm == 0x780 and field in v.WIDTH_FIELDS) or (imm == 0x438 and field - 4 in v.WIDTH_FIELDS) or \
            (imm == v.ASPECT_16_9 and field in (0x20, 0x2c)):
        sizes.append(f'({a:#x}, {reg:#x}, {field:#04x})')
    elif (field, imm) in ((0x30, 2560), (0x34, 1216), (0x40, 240), (0x44, 270)):
        scaled.append(f'({a:#x}, {reg:#x}, {field:#04x}, {imm}, {0 if field in (0x30, 0x40) else 1})')
print('SITES = [' + ', '.join(sizes) + ']')
print('SCALED = [' + ', '.join(scaled) + ']')
