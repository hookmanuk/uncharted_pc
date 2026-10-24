"""arena_owner.py: for each HeapArena arena_scan.py finds in GPU memory (0x11..), the qwords around it that hold its
size (the owner's whole size: the arena at owner+0x50 keeps it at +0x78)."""
import subprocess
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import memscan  # noqa: E402

out = subprocess.run([sys.executable, str(Path(__file__).with_name('arena_scan.py'))], capture_output=True, text=True).stdout
for line in out.splitlines():
    if 'memory 0x1' not in line:
        continue
    at = int(line.split('@')[1].split(':')[0], 16)
    size = float(line.split('size')[1].split('MB')[0]) * 2**20
    q = struct.unpack('<32Q', memscan.read(at - 0x50, 0x100))
    hits = [f'+{i * 8 - 0x50:#x}' for i, v in enumerate(q) if abs(v - size) < 2**20]
    print(line.split('used')[0].strip(), '| its size at arena', ' '.join(hits) or '-')
