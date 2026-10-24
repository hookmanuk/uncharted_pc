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
import os
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
# Past 4 GB: the 32 bits are HeapArena::Init's (rdi the arena, rsi its memory, edx its size: spilled and reloaded as a
# dword, so the arena's end and size were the context's size modulo 4 GB: "GPU Mem [F:243269632 T:897581056]" for a
# 5192548352-byte context). Its callers keep the whole size in the arena's owner (+0xc8, the arena at +0x50: the
# arena's +0x78), so Init takes the size from there when its low 32 bits are the ones passed, kept in a qword of the
# frame ([rbp-0x40], free; the stack protector's store gives the bytes, its check goes). Other arenas (the caller
# whose arena is at +0x10 of its owner) keep their 32-bit sizes: a mismatch, the passed size.
# (image offset of the size spill, of its reload, of the stack protector check's jne)
U1_ARENA_INIT = (0x35c8e1, 0x35c945, 0x35c998)
# The memory table (contexts first, then the arenas: 32-byte entries id, type, size, alignment) of each game, read by
# id: the CPU context (2, kAllocAppCpu) holds the "screenshoot buffer" arena, three full-screen RGBA buffers
# (3840 per eye: two of 56 MB took 112 of its 128 MB, the third failed).
U1_TABLE = 0x9713b0
U2_TABLE = 0xe6af10
U3_TABLE = 0x13a9ec0
CPU_CONTEXT_ID = 2
SCREENSHOT_ID = 0x80  # (0x81, the 128 MB before it, is another: growing it only moved this one along)
SCREENSHOT_BUFFERS = 3
U2_ARENA_INIT = (0x4e1f01, 0x4e1f65, 0x4e1fb8)
U3_ARENA_INIT = (0x4957c1, 0x495825, 0x495878)
ARENA_SPILL = (bytes.fromhex('8955c4 4989f6 4989fd 48'), bytes.fromhex('05'), 4, bytes.fromhex('488b00 488945d0'))
ARENA_SPILL_NEW = bytes.fromhex('4989f6 4989fd 89d0 488b4f78 39d1 480f44c1 488945c0 90')
ARENA_RELOAD = bytes.fromhex('8b45c4 498b4d18 488d148500000000')
ARENA_RELOAD_NEW = bytes.fromhex('488b45c0 498b4d18 4889c2 48c1e202')
ARENA_CHECK = bytes.fromhex('750f')
ARENA_CHECK_NEW = bytes.fromhex('6690')
# A heap past 4 GB: each arena's memory is reserved from its context's arena (the table's id: +0x498b30 in U3) with
# the whole size, which three steps cut to 32 bits: the reservation (mov esi, r12d before its allocator's virtual
# Allocate), that Allocate (mov ebx, esi: the arena owner's), and HeapArena::Allocate (mov r15d, esi). A 4161 MB
# render target heap took 65 MB of the GPU arena and the next arena (the texture allocator) began inside it:
# "Views into vram allocator not supported yet". Each step keeps the 64 bits (Allocate's callers may leave anything in
# the upper half of a 32-bit size: HeapArena::Allocate takes it only for 4-64 GB, upper 1..15; a guest pointer's is
# 0x10 and more). Found by their bytes (one match each); the stack protector's store gives HeapArena::Allocate's room.
ALLOC64 = [  # (pattern, offset in the match, original, new); the first: every match (the reservation's four
    # overloads, each taking a 64-bit size: mov r12, rdi)
    (rb'\x48\x89\xc7\x44\x89\xe6\x44\x89\xf2\x49\x89\xc9\xff\x53\x48', 3, '4489e6', '4c89e6'),
    (rb'\x44\x89\x6d\xcc\x89\xf3\x49\x89\xfe\xe8....\x85\xdb\xbe\x01\x00\x00\x00\x0f\x45\xf3', 0, '44896dcc89f3',
     '8955cc4889f3'),
    (rb'\x44\x89\x6d\xcc\x89\xf3\x49\x89\xfe\xe8....\x85\xdb\xbe\x01\x00\x00\x00\x0f\x45\xf3', 14,
     '85dbbe010000000f45f3', '4889de4885f67502ffc6'),
    (rb'\x41\x89\xf7\x49\x89\xfd\x48[\x8b\x8d]\x05....\x48\x8b\x00\x48\x89\x45\xd0\x44\x89\xf8\x48\x89\x85\xc8\xfe\xff\xff', 0,
     None, '4989fd 89f0 4889f1 48c1e920 83f910 7303 4889f0 4989c7 488985c8feffff'),
    (rb'\x45\x85\xff\xb8\x01\x00\x00\x00\x48\x8b\xbd\xc8\xfe\xff\xff\x48\x0f\x44\xf8', 0, '4585ff', '4d85ff'),
]
# HeapArena::Allocate's stack protector check (its store is gone): the first after its prologue.
ALLOC64_CHECK = rb'\x48[\x8b\x8d]\x0d....\x48\x8b\x09\x48\x3b\x4d\xd0\x75'

# The GPU context with the 64-bit arena: the direct memory and the render target heaps grow to the eye size.
GPU_CONTEXT_LIMIT_64 = (12288 - 2) << 20
if os.environ.get('VR_GPU_CONTEXT_MB'):  # development: another limit
    GPU_CONTEXT_LIMIT = GPU_CONTEXT_LIMIT_64 = int(os.environ['VR_GPU_CONTEXT_MB']) << 20
U1_CONTEXT = (0x9713f8, 0xbac00000)
# (Entries 1 + 2 + 3 make the system context, 4.5 GB; entry 2 is kAllocAppCpu, 1600 MB.)
# The system direct memory context that holds it (4.5 GB): an immediate, movabs rdx, 0x120000000 at image +0xcb.
U1_SYSTEM = (0xcb, 0x120000000)
DMEM_MB = 5056  # the retail PS4's direct memory (src/runtime_memory.c)
# Uncharted 2 (big2-ps4_Shipping.elf): the same configuration and tables at other addresses (the initialisers at
# +0x5525da, +0x5528c8 and +0x552b7e; arenas 0x84 / 0x85 at +0xe6bd30 / +0xe6bd50, spare 0xe 177 MB; the GPU
# context at +0xe6af50; the system context's movabs at +0xcb). Sites from tools/vr/res_sites.py.
U2_SITES = [
    (0x5525ec, 0x47, 0x08), (0x5525f3, 0x47, 0x0c), (0x552601, 0x47, 0x18), (0x552608, 0x47, 0x1c),
    (0x55260f, 0x47, 0x20), (0x552616, 0x47, 0x24), (0x55261d, 0x47, 0x28), (0x552624, 0x47, 0x2c),
    (0x55265c, 0x47, 0x38), (0x552663, 0x47, 0x3c), (0x552767, 0x47, 0x20), (0x55277c, 0x47, 0x2c),
    (0x55282f, 0x47, 0x20), (0x552844, 0x47, 0x2c), (0x5528dd, 0x47, 0x08), (0x5528e4, 0x47, 0x0c),
    (0x5528f2, 0x47, 0x18), (0x5528f9, 0x47, 0x1c), (0x552900, 0x47, 0x20), (0x552907, 0x47, 0x24),
    (0x55290e, 0x47, 0x28), (0x552915, 0x47, 0x2c), (0x5529c8, 0x47, 0x20), (0x5529dd, 0x47, 0x2c),
    (0x552b93, 0x40, 0x08), (0x552b9a, 0x40, 0x0c), (0x552ba8, 0x40, 0x18), (0x552baf, 0x40, 0x1c),
    (0x552bb6, 0x40, 0x20), (0x552bbd, 0x40, 0x24), (0x552bc4, 0x40, 0x28), (0x552bcb, 0x40, 0x2c),
    (0x552c03, 0x40, 0x38), (0x552c0a, 0x40, 0x3c),
]
U2_SCALED = [
    (0x552632, 0x47, 0x40, 240, 0), (0x552639, 0x47, 0x44, 270, 1), (0x55264e, 0x47, 0x30, 2560, 0),
    (0x552655, 0x47, 0x34, 1216, 1), (0x552923, 0x47, 0x40, 240, 0), (0x55292a, 0x47, 0x44, 270, 1),
    (0x552bd9, 0x40, 0x40, 240, 0), (0x552be0, 0x40, 0x44, 270, 1), (0x552bf5, 0x40, 0x30, 2560, 0),
    (0x552bfc, 0x40, 0x34, 1216, 1),
]
U2_HEAPS = [(0xe6bd38, 0x8bc4000), (0xe6bd58, 0x1f10bc00)]
U2_SPARE = (0xe6be38, 0xb0b0400)
U2_CONTEXT = (0xe6af58, 0xbac00000)
U2_SYSTEM = (0xcb, 0x120000000)
# Uncharted 3 (big3-ps4_Shipping.elf): the initialisers at +0x502f60, +0x503251 and +0x503597 (U2's, 0x4f660
# lower); the arenas in the RELRO segment (0x84 140 MB at +0x13aad20, 0x85 561 MB at +0x13aad40, spare 0xe 209 MB);
# the GPU context 3052 MB (+0x13a9f00); the system context's movabs at +0xcb.
U3_SITES = [
    (0x502f7c, 0x47, 0x08), (0x502f83, 0x47, 0x0c), (0x502f91, 0x47, 0x18), (0x502f98, 0x47, 0x1c),
    (0x502f9f, 0x47, 0x20), (0x502fa6, 0x47, 0x24), (0x502fad, 0x47, 0x28), (0x502fb4, 0x47, 0x2c),
    (0x502fec, 0x47, 0x38), (0x502ff3, 0x47, 0x3c), (0x5030f7, 0x47, 0x20), (0x50310c, 0x47, 0x2c),
    (0x5031bf, 0x47, 0x20), (0x5031d4, 0x47, 0x2c), (0x50326d, 0x47, 0x08), (0x503274, 0x47, 0x0c),
    (0x503282, 0x47, 0x18), (0x503289, 0x47, 0x1c), (0x503290, 0x47, 0x20), (0x503297, 0x47, 0x24),
    (0x50329e, 0x47, 0x28), (0x5032a5, 0x47, 0x2c), (0x503358, 0x47, 0x20), (0x50336d, 0x47, 0x2c),
    (0x5035b3, 0x40, 0x08), (0x5035ba, 0x40, 0x0c), (0x5035c8, 0x40, 0x18), (0x5035cf, 0x40, 0x1c),
    (0x5035d6, 0x40, 0x20), (0x5035dd, 0x40, 0x24), (0x5035e4, 0x40, 0x28), (0x5035eb, 0x40, 0x2c),
    (0x503623, 0x40, 0x38), (0x50362a, 0x40, 0x3c),
]
U3_SCALED = [
    (0x502fc2, 0x47, 0x40, 240, 0), (0x502fc9, 0x47, 0x44, 270, 1), (0x502fde, 0x47, 0x30, 2560, 0), (0x502fe5, 0x47, 0x34, 1216, 1),
    (0x5032b3, 0x47, 0x40, 240, 0), (0x5032ba, 0x47, 0x44, 270, 1), (0x5035f9, 0x40, 0x40, 240, 0), (0x503600, 0x40, 0x44, 270, 1),
    (0x503615, 0x40, 0x30, 2560, 0), (0x50361c, 0x40, 0x34, 1216, 1),
]
U3_HEAPS = [(0x13aad28, 0x8bc4000), (0x13aad48, 0x2310bc00)]
U3_SPARE = (0x13aae28, 0xd0b0400)
U3_CONTEXT = (0x13a9f08, 0xbec00000)
U3_SYSTEM = (0xcb, 0x120000000)
SITES = {'eboot.bin': (U1_SITES, U1_SCALED, (U1_HEAPS, U1_SPARE), (U1_CONTEXT,), U1_SYSTEM, U1_ARENA_INIT, U1_TABLE),
         'big2-ps4_Shipping.elf': (U2_SITES, U2_SCALED, (U2_HEAPS, U2_SPARE), (U2_CONTEXT,), U2_SYSTEM, U2_ARENA_INIT, U2_TABLE),
         'big3-ps4_Shipping.elf': (U3_SITES, U3_SCALED, (U3_HEAPS, U3_SPARE), (U3_CONTEXT,), U3_SYSTEM, U3_ARENA_INIT, U3_TABLE)}
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


def arena_patches(elf, arena):
    """HeapArena::Init with a 64-bit size (above), or None when the bytes are not the expected ones."""
    spill, reload, check = arena
    head, mid, gap, tail = ARENA_SPILL
    at = file_offset(elf, spill, len(ARENA_SPILL_NEW))
    if at is None or elf[at:at + len(head)] != head or elf[at + len(head) + 1:at + len(head) + 2] != mid or \
            elf[at + len(head)] not in (0x8b, 0x8d) or elf[at + len(head) + 2 + gap:at + len(ARENA_SPILL_NEW)] != tail:
        return None
    out = [(spill, ARENA_SPILL_NEW)]
    for address, original, new in ((reload, ARENA_RELOAD, ARENA_RELOAD_NEW), (check, ARENA_CHECK, ARENA_CHECK_NEW)):
        at = file_offset(elf, address, len(original))
        if at is None or elf[at:at + len(original)] != original:
            return None
        out.append((address, new))
    return out


def table_entry(elf, table, ident):
    """(image offset of the size, the size) of the memory table's entry `ident`, or None."""
    for n in range(512):
        at = file_offset(elf, table + n * 32, 32)
        if at is None:
            return None
        entry, kind, entry_size, _align = struct.unpack_from('<IIQQ', elf, at)
        if not (1 <= kind <= 12) or entry_size >> 40:
            return None  # past the table
        if entry == ident:
            return table + n * 32 + 8, entry_size
    return None


def code_segment(elf):
    phoff, = struct.unpack_from('<Q', elf, 0x20)
    phentsize, phnum = struct.unpack_from('<HH', elf, 0x36)
    for i in range(phnum):
        kind, flags, offset, vaddr, _paddr, filesz = struct.unpack_from('<IIQQQQ', elf, phoff + i * phentsize)
        if kind == 1 and flags & 1:
            return vaddr, elf[offset:offset + filesz]
    return 0, b''


def alloc64_patches(elf):
    """The 64-bit reservation of an arena (above), or None when a site is not found exactly once."""
    import re
    vaddr, code = code_segment(elf)
    out, prologue = [], None
    for n, (pattern, at, original, new) in enumerate(ALLOC64):
        matches = [m.start() for m in re.finditer(pattern, code, re.S)]
        if not matches or (n and len(matches) != 1):
            return None
        new = bytes.fromhex(new)
        for match in matches:
            start = match + at
            if original is not None and code[start:start + len(new)] != bytes.fromhex(original):
                return None
            if original is None:
                prologue = start
            out.append((vaddr + start, new))
    check = re.search(ALLOC64_CHECK, code[prologue:prologue + 0x200], re.S)
    if not check or code[prologue + check.end():prologue + check.end() + 1] not in (b'\x0f', b'\x10', b'\x11', b'\x12',
                                                                                b'\x13', b'\x14', b'\x15'):
        return None
    out.append((vaddr + prologue + check.end() - 1, b'\x66\x90'))
    return out


def patches(elf, sites, size, fields):
    sites, scaled, (heaps, spare), context, system, arena, table = sites
    out = []
    for address, original in [*heaps, spare, *context]:
        at = file_offset(elf, address, 8)
        if at is None or struct.unpack_from('<Q', elf, at)[0] != original:
            return None
    # The heaps' growth comes with the context's (below 4 GB without the arena fix). Whole 2 MB: the game
    # asserts its memory contexts are (+0x36a430: megabytes, +0x36a8c3: even ones).
    pixels = size * size / (1920 * 1080)
    arena_fix = arena_patches(elf, arena)
    budget = (GPU_CONTEXT_LIMIT_64 if arena_fix else GPU_CONTEXT_LIMIT) - context[0][1]
    patches.capped = False
    growths = []
    for address, original in heaps:
        wanted = (max(0, int(original * pixels * 1.02) - original) + 0x1fffff) & ~0x1fffff
        growth = min(wanted, budget & ~0x1fffff)
        patches.capped |= growth < wanted
        budget -= growth
        growths.append(growth)
        out.append((address, struct.pack('<Q', original + growth)))
        if original + growth > 0xffffffff:  # a heap past 4 GB: its reservation in 64 bits
            alloc64 = alloc64_patches(elf)
            if alloc64 is None:
                return None
            out += [patch for patch in alloc64 if patch not in out]
    extra = sum(growths)
    for address, original in context:
        at = file_offset(elf, address, 8)
        if at is None or struct.unpack_from('<Q', elf, at)[0] != original:
            return None
        out.append((address, struct.pack('<Q', original + extra)))
        if original + extra > 0xffffffff:  # the GPU Mem arena's size in 64 bits
            out += arena_fix
    patches.arena_64 = any(original + extra > 0xffffffff for _, original in context)
    # The screenshot arena (and its CPU context) for three buffers of the eye size, plus 8 MB.
    shot, cpu = table_entry(elf, table, SCREENSHOT_ID), table_entry(elf, table, CPU_CONTEXT_ID)
    if shot is None or cpu is None:
        return None
    shot_growth = (max(0, SCREENSHOT_BUFFERS * size * size * 4 + (8 << 20) - shot[1]) + 0x1fffff) & ~0x1fffff
    if shot_growth:
        out.append((shot[0], struct.pack('<Q', shot[1] + shot_growth)))
        out.append((cpu[0], struct.pack('<Q', cpu[1] + shot_growth)))
    address, original = system
    expected = bytes.fromhex('48ba') + struct.pack('<Q', original)
    at = file_offset(elf, address, len(expected))
    if at is None or elf[at:at + len(expected)] != expected:
        return None
    out.append((address + 2, struct.pack('<Q', original + extra * len(context) + shot_growth)))
    patches.extra_mb = (extra * len(context) + shot_growth + 0xfffff) >> 20
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
