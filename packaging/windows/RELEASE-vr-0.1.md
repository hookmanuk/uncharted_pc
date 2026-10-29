# vr-0.1: Uncharted in VR (OpenXR)

**Uncharted 1 is playable.**
**Uncharted 2 & 3 are untested and likely broken.**

Download `uncharted-windows.zip`, unzip it anywhere and run `Uncharted.exe`. Nothing else to install.

1. Game page: choose the folder with your own decrypted dump of the collection (no game files are included).
2. Display page: tick **Play in a VR headset (OpenXR)**. Start your OpenXR runtime first (SteamVR, Virtual Desktop,
   Meta Link, Windows Mixed Reality).
3. Play page: pick Drake's Fortune and press **PLAY**. F8 recentres.

- Uncharted 1: 3D with head tracking, HUD on its own layer, the headset's own refresh rate.
  NVIDIA DLSS (GeForce RTX) is optional and off by default.
  Resolution per eye on the Display page (start at 2160).
- Menus and cutscene videos show on a large screen in front of you.
- The first run builds shaders and can stutter or glitch; run again.
- Needs Windows 10 (1903+) or 11, a Vulkan 1.3 GPU and about 8 GB of free memory.

Based on drakolordx7/uncharted_pc (community edition of SuperMedo's port).
