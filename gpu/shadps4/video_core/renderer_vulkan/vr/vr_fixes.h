// SPDX-License-Identifier: GPL-2.0-or-later
// VR (dcport-vr): per-game fixes to shaders for render sizes other than the game's own (the VR render is square, e.g.
// 2880x2880), and the game's render and output sizes. Uncharted fork: the collection's 1920x1080 (BB_RENDER_RES), no
// shader fixes yet; Driveclub's fix (compute 0x2d9c66a5's 960x540 bounds) is in the Driveclub recomp's vr branch.

#pragma once

#include <vector>

#include "common/types.h"
#include "shader_recompiler/info.h"

namespace Vulkan::Vr::Fixes {

/// Pipeline cache, after a guest shader's SPIR-V is made: rewrites constants the game's shaders hold for its own
/// render size.
void PatchShader(u64 hash, std::vector<u32>& spirv);

/// The game's render size; false while unknown.
bool RenderSize(u32& width, u32& height);

/// The game's render size is `width` x `height` (1920x1080 outside VR): the port's scene detection (VR: square sizes).
inline bool NativeSize(u32 width, u32 height) {
    u32 w = 0, h = 0;
    return RenderSize(w, h) ? width == w && height == h : width == 1920 && height == 1080;
}

/// The game's output size (the display buffer; above the render size when an upscaler runs); false while unknown.
bool OutputSize(u32& width, u32& height);

/// The video decoder (Uncharted fork): a movie frame in NV12 at `buffer` (luma rows `pitch` bytes apart, `rows` of
/// them, the chroma after them), `width` x `height` pixels.
void NoteMovieFrame(u64 buffer, u32 pitch, u32 width, u32 height, u32 rows);

/// Each dispatch (Uncharted fork): the collection's movie shader (NV12 to the display buffer) gets its luma and chroma
/// images at the render size (the game makes them from its display configuration, square in VR: rows sheared, the
/// chroma read from the next frame's buffer); they become the decoded frame's layout.
void FixMovieImages(const Shader::Info& cs);

/// The video decoder, VR (Uncharted fork): the canvas movies are fitted into (black bars, the movie's shape kept), as
/// the community build's BB_GAME_RES display size: the render size's shape in a 1080p frame's buffer (1408 x 1408 for a
/// square). The movie shader stretches the canvas over the display buffer (vk_shader_util.cpp), so a movie keeps its
/// shape on the headset's screen at the canvas's aspect, as the menus drawn over it do. False outside VR (`width`,
/// `height` untouched).
bool MovieCanvas(s32& width, s32& height);

/// A movie plays (its shader ran in the last 0.3 s): its size. It fills the display buffer (MovieCanvas): the headset
/// shows the whole frame on the screen, not stereo.
bool MovieRect(u32& width, u32& height);

} // namespace Vulkan::Vr::Fixes
