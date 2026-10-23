#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""VR fork: the games' render size for VR (BB_VR_EYE_RES, square). The collection sets up its display configuration
with immediates (mov dword [reg+field], 1920 / 1080 / 16:9 as a float); those become the eye size, so every render
target is made at that size from the start (multiview gives each a second layer for the right eye).

    vr_resolution.py OUT_DIR EXE SIZE [FIELDS]   writes OUT_DIR/vr_patches.bin (BBPATCH2) or removes it

Each site's original instruction is checked in OUT_DIR/eboot.elf (prepare.py's plaintext ELF; image offsets are its
virtual addresses): a game or version whose bytes differ gets no patch. FIELDS: the configuration's fields to change
(hex offsets, comma-separated, 1000 for all the size pairs; default all of them, the scaled ones too).
"""
import struct
import sys
from pathlib import Path

# Uncharted 1 (eboot.bin, CUSA02343 01.00): the display configuration (image +0xd5c2fc) initialisers at
# +0x3c4d60, +0x3c5050 and +0x3c5300. Fields: +0x08 / +0x18 / +0x24 / +0x38 width and height pairs, +0x20 / +0x2c their
# aspect (16:9). (+0x30 2560x1216 and +0x40 240x270 are left alone.) Sites: (image offset, register byte, field).
U1_SITES = [
    (0x3c4d7c, 0x47, 0x08), (0x3c4d83, 0x47, 0x0c), (0x3c4d91, 0x47, 0x18), (0x3c4d98, 0x47, 0x1c),
    (0x3c4d9f, 0x47, 0x20), (0x3c4da6, 0x47, 0x24), (0x3c4dad, 0x47, 0x28), (0x3c4db4, 0x47, 0x2c),
    (0x3c4dec, 0x47, 0x38), (0x3c4df3, 0x47, 0x3c), (0x3c4ef7, 0x47, 0x20), (0x3c4f0c, 0x47, 0x2c),
    (0x3c4fbf, 0x47, 0x20), (0x3c4fd4, 0x47, 0x2c), (0x3c506d, 0x47, 0x08), (0x3c5074, 0x47, 0x0c),
    (0x3c5082, 0x47, 0x18), (0x3c5089, 0x47, 0x1c), (0x3c5090, 0x47, 0x20), (0x3c5097, 0x47, 0x24),
    (0x3c509e, 0x47, 0x28), (0x3c50a5, 0x47, 0x2c), (0x3c5158, 0x47, 0x20), (0x3c516d, 0x47, 0x2c),
    (0x3c5323, 0x40, 0x08), (0x3c532a, 0x40, 0x0c), (0x3c5338, 0x40, 0x18), (0x3c533f, 0x40, 0x1c),
    (0x3c5346, 0x40, 0x20), (0x3c534d, 0x40, 0x24), (0x3c5354, 0x40, 0x28), (0x3c535b, 0x40, 0x2c),
    (0x3c5393, 0x40, 0x38), (0x3c539a, 0x40, 0x3c),
]
# Sizes derived from 1920x1080, scaled with it: (image offset, register byte, field, original value, axis 0 x / 1 y).
U1_SCALED = [
    (0x3c4dde, 0x47, 0x30, 2560, 0), (0x3c4de5, 0x47, 0x34, 1216, 1), (0x3c5385, 0x40, 0x30, 2560, 0),
    (0x3c538c, 0x40, 0x34, 1216, 1), (0x3c4dc2, 0x47, 0x40, 240, 0), (0x3c4dc9, 0x47, 0x44, 270, 1),
    (0x3c50b3, 0x47, 0x40, 240, 0), (0x3c50ba, 0x47, 0x44, 270, 1), (0x3c5369, 0x40, 0x40, 240, 0),
    (0x3c5370, 0x40, 0x44, 270, 1),
]
# The memory arenas' table (image +0x972100.., 32-byte entries: id, type, size, alignment): entry 0x84 is the render
# targets' heap ("Render Target", 0x8bc4000 = 140 MB for 1920x1080); it grows with the pixel count (2880x2880 ran out:
# "HeapArena::Allocate: out of memory ... Render Target"). (image offset of the u64 size, original size)
# The GPU context's size grows by the same amount (the memory contexts' table, image +0x9713b0.., entry 3
# kAllocAppGpu 0xbac00000: the game checks that its arenas add up to it), and the direct memory with it (run.py:
# BB_DMEM_MB, the runtime's sceKernelGetDirectMemorySize).
# Entry 0x85 is the second ("Render Target Heap", 0x1f10bc00 = 497 MB), full at 2880x2880 too. Entry 0xe is the
# context's spare ("ALLOCATION_APP_GPU_FREE", 0xd0b0400 = 209 MB): checked, left alone (context 7's arenas need it:
# taking it, their 144 MB arena no longer fitted).
U1_HEAPS = [(0x9721d8, 0x8bc4000), (0x9721f8, 0x1f10bc00)]
U1_SPARE = (0x9722d8, 0xd0b0400)
# The GPU context must stay below 4 GB: the game keeps its "GPU Mem" arena's size in 32 bits (5028 MB came out as
# 932 MB, and the heaps no longer fitted). Growth past that is not given (larger eyes: an upscaler from a smaller
# render).
GPU_CONTEXT_LIMIT = (4096 - 2) << 20
U1_CONTEXT = (0x9713f8, 0xbac00000)
# (Entries 1 + 2 + 3 make the system context, 4.5 GB; entry 2 is kAllocAppCpu, 1600 MB.)
# The system direct memory context that holds it (4.5 GB): an immediate, movabs rdx, 0x120000000 at image +0xcb.
U1_SYSTEM = (0xcb, 0x120000000)
DMEM_MB = 5056  # the retail PS4's direct memory (src/runtime_memory.c)
SITES = {'eboot.bin': (U1_SITES, U1_SCALED, (U1_HEAPS, U1_SPARE), (U1_CONTEXT,), U1_SYSTEM)}
ASPECT_16_9 = 0x3fe38e39
WIDTH_FIELDS = {0x08, 0x18, 0x24, 0x38}


def segments(elf):
    if elf[:4] != b'\x7fELF':
        raise ValueError('not an ELF')
    phoff, = struct.unpack_from('<Q', elf, 0x20)
    size, count = struct.unpack_from('<HH', elf, 0x36)
    for i in range(count):
        kind, _flags, offset, vaddr, _paddr, filesz, _memsz, _align = struct.unpack_from('<IIQQQQQQ', elf, phoff + i * size)
        if kind in (1, 0x61000010) and filesz:
            yield vaddr, offset, filesz


def file_offset(elf, address, length):
    for vaddr, offset, filesz in segments(elf):
        if vaddr <= address and address + length <= vaddr + filesz:
            return offset + address - vaddr
    return None


def patches(elf, sites, size, fields):
    sites, scaled, (heaps, spare), context, system = sites
    out = []
    for address, original in [*heaps, spare, *context]:
        at = file_offset(elf, address, 8)
        if at is None or struct.unpack_from('<Q', elf, at)[0] != original:
            return None
    # The heaps' growth comes with the context's (below its 4 GB limit). Whole 2 MB: the game
    # asserts its memory contexts are (+0x36a430: megabytes, +0x36a8c3: even ones).
    pixels = size * size / (1920 * 1080)
    budget = GPU_CONTEXT_LIMIT - context[0][1]
    patches.capped = False
    growths = []
    for address, original in heaps:
        wanted = (max(0, int(original * pixels * 1.02) - original) + 0x1fffff) & ~0x1fffff
        growth = min(wanted, budget & ~0x1fffff)
        patches.capped |= growth < wanted
        budget -= growth
        growths.append(growth)
        out.append((address, struct.pack('<Q', original + growth)))
    extra = sum(growths)
    for address, original in context:
        at = file_offset(elf, address, 8)
        if at is None or struct.unpack_from('<Q', elf, at)[0] != original:
            return None
        out.append((address, struct.pack('<Q', original + extra)))
    address, original = system
    expected = bytes.fromhex('48ba') + struct.pack('<Q', original)
    at = file_offset(elf, address, len(expected))
    if at is None or elf[at:at + len(expected)] != expected:
        return None
    out.append((address + 2, struct.pack('<Q', original + extra * len(context))))
    patches.extra_mb = (extra * len(context) + 0xfffff) >> 20
    for address, reg, field, original, axis in scaled:
        if fields and field not in fields:
            continue
        expected = bytes([0xc7, reg, field]) + struct.pack('<I', original)
        at = file_offset(elf, address, len(expected))
        if at is None or elf[at:at + len(expected)] != expected:
            return None
        out.append((address + 3, struct.pack('<I', round(original * size / (1920 if axis == 0 else 1080)))))
    for address, reg, field in sites:
        width_field = field if field in WIDTH_FIELDS else field - 4 if field - 4 in WIDTH_FIELDS else field - 8
        if fields and width_field not in fields and not fields & {0x1000}:
            continue
        original = 0x780 if field in WIDTH_FIELDS else 0x438 if field - 4 in WIDTH_FIELDS else ASPECT_16_9
        expected = bytes([0xc7, reg, field]) + struct.pack('<I', original)
        at = file_offset(elf, address, len(expected))
        if at is None or elf[at:at + len(expected)] != expected:
            return None  # another game or version
        value = struct.unpack('<I', struct.pack('<f', 1.0))[0] if original == ASPECT_16_9 else size
        out.append((address + 3, struct.pack('<I', value)))
    return out


def main():
    out_dir, exe, size = Path(sys.argv[1]), Path(sys.argv[2]).name, int(sys.argv[3] or 0)
    fields = {int(f, 16) for f in sys.argv[4].split(',')} if len(sys.argv) > 4 and sys.argv[4] else set()
    target = out_dir / 'vr_patches.bin'
    target.unlink(missing_ok=True)
    (out_dir / 'vr_patches.dmem').unlink(missing_ok=True)
    sites = SITES.get(exe)
    elf_path = out_dir / 'eboot.elf'
    if not size or not sites or not elf_path.is_file():
        return 0
    found = patches(elf_path.read_bytes(), sites, size, fields)
    if found is None:
        print(f'VR: {exe} is not the version the render size patch knows: 1920x1080 kept')
        return 0
    with target.open('wb') as f:
        f.write(b'BBPATCH2' + struct.pack('<QQ', 0x400000, len(found)))
        for offset, data in found:
            f.write(struct.pack('<QQ', offset, len(data)) + data)
    (out_dir / 'vr_patches.dmem').write_text(str(DMEM_MB + patches.extra_mb + 64), encoding='ascii')
    print(f'VR: render size {size}x{size} ({len(found)} patches, {patches.extra_mb} MB more GPU memory'
          f'{", the render target heaps at the 4 GB limit" if patches.capped else ""})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
