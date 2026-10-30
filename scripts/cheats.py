"""Compile the enabled cheats of one executable into cheats.bin for bb-probe --patches.

Cheats are "health" and "ammo". The executables of the versions listed below (by SHA-256) are patched at their known
addresses when the bytes there are the expected ones. Any other version is searched for each cheat's code
(SIGNATURES, made from the 1.00 collection with tools/vr/sigmake.py): a cheat is written only when its pattern is
found exactly once. Anything else is reported and skipped, never an error.
Addresses are image offsets (virtual addresses of the SELF's load segments).
"""
import hashlib
import re
from pathlib import Path
import struct

import prepare

NOP = '90'
V100 = {'eboot.bin': '8953b6ed5cf4d1d26aa3aa4cd1e0c72817ea155f6a5774fb17f6fec7d787f953',
        'big2-ps4_Shipping.elf': '61c0456314a8de2dbd32e1c426f7828092de7d25a58b83386ced1a48d7765ebe',
        'big3-ps4_Shipping.elf': 'acc952e5a9ac7062f8ecfc52700faa599d5f29deb1e24f9e81e6de832326d1d0'}
V102 = {'eboot.bin': 'dc017b45da8978bb11a0a11d940e96d98fa07f3c066a3debba043714432212f8',
        'big2-ps4_Shipping.elf': '5047b693b1c6fd06ccb1eacecacee60b6183bedc18d023cac276e1ec6009dfe2',
        'big3-ps4_Shipping.elf': '02c111aa7e73e445d71a1ad861702811f31d471abc61caf01732d55a77eb1c38'}

# sha256 -> cheat -> (address, original bytes, replacement)
PATCHES = {
    # health: jump to the engine's own no-damage path. ammo: the clip decrement is shared with NPC weapons, so U1/U2
    # drop the player Fire's call to it and U3 forces its player-held branch.
    V100['eboot.bin']: {'health': (0x261edb, '7535', 'eb35'),
                        'ammo': (0x71b003, 'e8c8acc2ff', NOP * 5)},
    V102['eboot.bin']: {'health': (0x26244b, '7535', 'eb35'),
                        'ammo': (0x730f73, 'e8c854c1ff', NOP * 5)},
    V100['big2-ps4_Shipping.elf']: {'health': (0x35de73, '0f84bc010000', 'e9cf02000090'),
                                    'ammo': (0xa4fd8d, 'e87e8aa6ff', NOP * 5)},
    V100['big3-ps4_Shipping.elf']: {'health': (0x33f87b, '0f84ca020000', 'e96a0b000090'),
                                    'ammo': (0x46a1f9, '4c8d3dc0660501', '41b401eb729090')},
    V102['big2-ps4_Shipping.elf']: {'health': (0x7f0143, '0f843c050000', 'e95f06000090'),
                                    'ammo': (0x961ac8, 'e853851800', NOP * 5)},
    V102['big3-ps4_Shipping.elf']: {'health': (0x9f9295, '4180bfa7000000007436', 'e9320300009090909090'),
                                    'ammo': (0xd517ac, '4c8d257d92aa0066418b442462', '40b701e9110100009090909090')},
}


# Version-independent cheats: exe -> cheat -> (pattern, offset of the patched bytes in it, replacement). A pattern is
# hex bytes with "??" where the code moves (call targets, rip-relative displacements). The replacement is hex bytes or
# ('jmp', pattern of the jump's destination, length): a jmp rel32 to where that pattern is, padded with nops.
SIGNATURES = {
    'eboot.bin': {
        'health': ('0a 11 75 35 85 c0 75 31 c4 c1 7a 10 86 28 14 00 00', 2, 'eb35'),
        'ammo': ('e8 ?? ?? ?? ?? 31 f6 4c 89 f7 e8 ?? ?? ?? ?? 49 8b 04 24 48 3b 45 d8', 10, NOP * 5)},
    'big2-ps4_Shipping.elf': {
        'health': ('80 7b 67 00 0f 84 bc 01 00 00 e8 ?? ?? ?? ?? 84 c0', 4,
                   ('jmp', '48 8b bd c8 58 ff ff 48 8b 07 ff 50 18 88 c3 88 9d c8 58 ff ff', 6)),
        'ammo': ('74 12 31 f6 4c 89 f7 e8 ?? ?? ?? ?? 4c 89 f7 e8 ?? ?? ?? ??', 7, NOP * 5)},
    'big3-ps4_Shipping.elf': {
        'health': ('41 80 bf a7 00 00 00 00 0f 84 ca 02 00 00 e8 ?? ?? ?? ?? 84 c0', 8,
                   ('jmp', '49 8b 04 24 4c 89 e7 ff 50 18 41 88 c4 41 80 bd e8 0c 00 00 00', 6)),
        'ammo': ('75 46 4c 8d 3d ?? ?? ?? ?? 66 41 8b 47 62 84 c0', 2, '41b401eb729090')},
}


def find_code(elf, program_headers, pattern):
    """Virtual addresses of every match of a masked hex pattern in the load segments."""
    regex = re.compile(b''.join(b'.' if t == '??' else re.escape(bytes.fromhex(t)) for t in pattern.split()), re.S)
    out = []
    for p in program_headers:
        if p['type'] in (1, 0x61000010) and p['filesz']:
            segment = bytes(elf[p['offset']:p['offset'] + p['filesz']])
            out += [p['vaddr'] + m.start() for m in regex.finditer(segment)]
    return out


def signature_write(elf, program_headers, name, signature):
    """(address, replacement) of one cheat found by its pattern, or None."""
    pattern, offset, replacement = signature
    found = find_code(elf, program_headers, pattern)
    if len(found) != 1:
        return None
    address = found[0] + offset
    if isinstance(replacement, tuple):
        _kind, destination_pattern, length = replacement
        destination = find_code(elf, program_headers, destination_pattern)
        if len(destination) != 1:
            return None
        replacement = b'\xe9' + struct.pack('<i', destination[0] - (address + 5)) + b'\x90' * (length - 5)
    else:
        replacement = bytes.fromhex(replacement)
    return address, replacement


def image_bytes(elf, program_headers, address, size):
    for p in program_headers:
        if p['type'] in (1, 0x61000010) and p['vaddr'] <= address and address + size <= p['vaddr'] + p['filesz']:
            start = p['offset'] + address - p['vaddr']
            return bytes(elf[start:start + size])
    return None


def compile_cheats(path, cheats):
    """[(address, replacement bytes)] for the enabled cheats; reasons go to stdout."""
    data = Path(path).read_bytes()
    table = PATCHES.get(hashlib.sha256(data).hexdigest())
    elf, _header, program_headers, _segments, _missing = prepare.parse_self(data)
    writes = []
    if table is None:  # another version: by the code
        signatures = SIGNATURES.get(Path(path).name, {})
        for name in cheats:
            if name not in signatures:
                continue
            write = signature_write(elf, program_headers, name, signatures[name])
            if write is None:
                print(f'Cheats: {name} skipped, its code was not found in {Path(path).name}')
                continue
            writes.append(write)
            print(f'Cheats: {name} on (found by its code)')
        return writes
    for name in cheats:
        if name not in table:
            continue
        address, original, replacement = table[name]
        original, replacement = bytes.fromhex(original), bytes.fromhex(replacement)
        if image_bytes(elf, program_headers, address, len(original)) != original:
            print(f'Cheats: {name} skipped, unexpected code at {address:#x}')
            continue
        writes.append((address, replacement))
        print(f'Cheats: {name} on')
    return writes


def write_patch_file(out, writes):
    with open(out, 'wb') as f:
        f.write(b'BBPATCH2' + struct.pack('<QQ', 0, len(writes)))
        for address, data in writes:
            f.write(struct.pack('<QQ', address, len(data)) + data)


def build(path, out_dir, cheats):
    """Path of out_dir/cheats.bin for the enabled cheats, or None when nothing applies."""
    out = Path(out_dir) / 'cheats.bin'
    try:
        out.unlink(missing_ok=True)
        writes = compile_cheats(path, cheats) if cheats else []
        if writes:
            write_patch_file(out, writes)
            return out
    except (OSError, ValueError, KeyError, IndexError) as error:
        print(f'Cheats: not applied ({error})')
    return None
