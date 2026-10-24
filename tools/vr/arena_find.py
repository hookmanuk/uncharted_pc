"""arena_find.py ELF: HeapArena::Init (the 32-bit size spill and reload) and its callers' size field in a game ELF."""
import re
import struct
import sys
from pathlib import Path

elf = Path(sys.argv[1]).read_bytes()
phoff, = struct.unpack_from('<Q', elf, 0x20)
phentsize, phnum = struct.unpack_from('<HH', elf, 0x36)
for i in range(phnum):
    p_type, p_flags, p_offset, p_vaddr, _, p_filesz = struct.unpack_from('<IIQQQQ', elf, phoff + i * phentsize)
    if p_type != 1 or not (p_flags & 1):
        continue
    code = elf[p_offset:p_offset + p_filesz]
    entry = rb'\x89\x55\xc4\x49\x89\xf6\x49\x89\xfd\x48[\x8b\x8d]\x05....\x48\x8b\x00\x48\x89\x45\xd0\x49\x8d\x5d\x30'
    reload = rb'\x8b\x45\xc4\x49\x8b\x4d\x18\x48\x8d\x14\x85\x00\x00\x00\x00\x48\x83\xe1\x03\x48\x09\xd1\x49\x89\x4d\x18'
    for m in re.finditer(entry, code, re.S):
        a = p_vaddr + m.start()
        r = re.search(reload, code[m.start():m.start() + 0x100], re.S)
        c = re.search(rb'\x48[\x8b\x8d]\x05....\x48\x8b\x00\x48\x3b\x45\xd0\x75', code[m.start() + 0x40:m.start() + 0x100], re.S)
        print(f'entry spill +{a:#x}, reload +{a + r.start():#x}' if r else f'entry +{a:#x} (no reload)',
              f'canary jne +{a + 0x40 + c.end() - 1:#x}' if c else '(no canary check)')
