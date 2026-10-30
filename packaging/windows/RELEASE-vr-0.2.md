# vr-0.2: Uncharted in VR (OpenXR)

**Uncharted 1 is playable.**
**Uncharted 2 & 3 are untested and likely broken.**

Download `uncharted-windows.zip`, unzip it anywhere and run `Uncharted.exe`. Nothing else to install.

1. Game page: choose the folder with your own decrypted dump of the collection (no game files are included).
2. Start your OpenXR runtime first (SteamVR, Virtual Desktop, Meta Link, Windows Mixed Reality). VR is ticked by
   default (Display page: **Play in a VR headset (OpenXR)**; untick it to play on the screen).
3. Play page: pick Drake's Fortune and press **PLAY**. F8 recentres.

New in vr-0.2:
- **Head aiming** (on by default): hold L2 and Nathan aims where you look, straight away, with the reticle on it.
  The right stick still turns you left and right. The thing you were looking at stays under your gaze while the
  camera moves to the shoulder; letting go returns the view without a jump. Uncharted 1 (2 and 3 untested).
- **Lock the vertical axis** (Never / Always / When aiming; Always by default): the headset looks up and down.
- Uncharted 1: on-screen prompts that were missing in VR now show (e.g. the boat's "Press Y after the first punch
  lands").
- Uncharted 2 & 3 (still untested): fixes for lights and mist that moved with the head, depth of field off in VR.
- Shader stutter: one shader cache for every resolution per eye, and updates no longer throw the whole cache away.
- Game versions are found by their code: a version the VR layer does not know says so, with a report to copy.

- Menus and cutscene videos show on a large screen in front of you.
- The first run builds shaders and can stutter or glitch; run again.
- Needs Windows 10 (1903+) or 11, a Vulkan 1.3 GPU and about 8 GB of free memory.

Based on drakolordx7/uncharted_pc (community edition of SuperMedo's port).
