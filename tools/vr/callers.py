"""callers.py ELF TARGET: direct calls and jumps (E8/E9 rel32) to TARGET (hex) in a plaintext game ELF's code, with the
return address of each call."""
import struct
import sys
from pathlib import Path

elf = Path(sys.argv[1]).read_bytes()
target = int(sys.argv[2], 16)
phoff, = struct.unpack_from('<Q', elf, 0x20)
phentsize, phnum = struct.unpack_from('<HH', elf, 0x36)
for i in range(phnum):
    p_type, p_flags, p_offset, p_vaddr, _, p_filesz = struct.unpack_from('<IIQQQQ', elf, phoff + i * phentsize)
    if p_type != 1 or not (p_flags & 1):
        continue
    code = elf[p_offset:p_offset + p_filesz]
    for at in range(len(code) - 5):
        if code[at] in (0xe8, 0xe9):
            rel, = struct.unpack_from('<i', code, at + 1)
            if p_vaddr + at + 5 + rel == target:
                print(f"+{p_vaddr + at:#x}: {'call' if code[at] == 0xe8 else 'jmp'} (returns to +{p_vaddr + at + 5:#x})")
