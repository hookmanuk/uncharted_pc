// SPDX-License-Identifier: GPL-2.0-or-later
// VR (dcport-vr): per-game fixes to shaders for render sizes other than the game's own (the VR render is square, e.g.
// 2880x2880), and the game's render and output sizes. Uncharted fork: the collection's 1920x1080 (BB_RENDER_RES), no
// shader fixes yet; Driveclub's fix (compute 0x2d9c66a5's 960x540 bounds) is in the Driveclub recomp's vr branch.

#pragma once

#include <vector>

#include "common/types.h"

namespace Vulkan::Vr::Fixes {

/// Pipeline cache, after a guest shader's SPIR-V is made: rewrites constants the game's shaders hold for its own
/// render size.
void PatchShader(u64 hash, std::vector<u32>& spirv);

/// The game's render size; false while unknown.
bool RenderSize(u32& width, u32& height);

/// The game's output size (the display buffer; above the render size when an upscaler runs); false while unknown.
bool OutputSize(u32& width, u32& height);

} // namespace Vulkan::Vr::Fixes
