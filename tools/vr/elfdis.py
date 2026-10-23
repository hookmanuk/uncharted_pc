"""elfdis.py ELF ADDR [BEFORE] [AFTER]: x86-64 disassembly of a plaintext game ELF (image offsets = virtual addresses)."""
import sys
from pathlib import Path
import capstone
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import vr_resolution as v

elf = Path(sys.argv[1]).read_bytes()
a = int(sys.argv[2], 16)
before = int(sys.argv[3], 0) if len(sys.argv) > 3 else 0x60
after = int(sys.argv[4], 0) if len(sys.argv) > 4 else 0x40
at = v.file_offset(elf, a - before, before + after)
md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
md.skipdata = True
for ins in md.disasm(elf[at:at + before + after], a - before):
    print(f"{'>>' if ins.address == a else '  '}+{ins.address:#x}: {ins.mnemonic} {ins.op_str}")
