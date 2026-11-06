# vr-0.3: Uncharted in VR (OpenXR), frame pacing

**This release is frame pacing fixes for Uncharted 1.**
**Uncharted 2 & 3 are still not ready.**

Download `uncharted-windows.zip`, unzip it anywhere and run `Uncharted.exe`. Nothing else to install.

1. Game page: choose the folder with your own decrypted dump of the collection (no game files are included).
2. Start your OpenXR runtime first (SteamVR, Virtual Desktop, Meta Link, Windows Mixed Reality). VR is ticked by
   default (Display page: **Play in a VR headset (OpenXR)**; untick it to play on the screen).
3. Play page: pick Drake's Fortune and press **PLAY**. F8 recentres.

New in vr-0.3:
- **Frame pacing:** the game now renders one frame for each frame the headset asks for. Before, it could render far
  more frames than the headset showed, and switched its frame cap on and off every few seconds when it fell behind.
- **Menus and play:** the switch between the flat menu screen and 3D now reads the game's own menu state in
  Uncharted 1 and 2, instead of guessing from a still camera.
- **Copy VR report** (Log page) only says the game version is unknown when it really is.
- **`uncharted-debug.bat`**: starts the launcher with timing logs on, with other test switches inside the file.
- Based on the latest community edition.

If you stream to a Quest and still get long stutters: we could not test this on a Quest ourselves. Please run
`uncharted-debug.bat`, play until it happens, and send us `user\last_run.log`, with your headset and how it connects.

Known issues:
- Uncharted 2: the world can jump when you turn your head, and shadows flicker.
- Uncharted 3: menus can still briefly show in 3D.
- DLSS can crash when the game loads on some GPUs (seen on an RTX 3070).
- Uncharted 1: one report of a crash while loading the temple area (an RTX 4090 at 2880 per eye).

- Menus and cutscene videos show on a large screen in front of you.
- The first run builds shaders and can stutter or glitch; run again.
- Needs Windows 10 (1903+) or 11, a Vulkan 1.3 GPU and about 8 GB of free memory.

Based on drakolordx7/uncharted_pc (community edition of SuperMedo's port).
