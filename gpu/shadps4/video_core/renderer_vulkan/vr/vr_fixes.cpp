// SPDX-License-Identifier: GPL-2.0-or-later
// VR (dcport-vr): see vr_fixes.h.

#include <cstdio>
#include <cstdlib>

#include "video_core/renderer_vulkan/vr/vr_fixes.h"

namespace Vulkan::Vr::Fixes {
namespace {

[[maybe_unused]] // the per-game fixes to come
// Every OpConstant of a 32-bit type holding `from` becomes `to`; the number changed.
int ReplaceConstant(std::vector<u32>& spirv, u32 from, u32 to) {
    int changed = 0;
    for (size_t i = 5; i < spirv.size();) {
        const u32 words = spirv[i] >> 16, op = spirv[i] & 0xffff;
        if (words == 0) {
            break;
        }
        if (op == 43 /* OpConstant */ && words == 4 && spirv[i + 3] == from) {
            spirv[i + 3] = to;
            ++changed;
        }
        i += words;
    }
    return changed;
}

} // namespace

// Uncharted fork: the games' render and output sizes are not known yet (stage 2 finds them); until then no image
// is a main-view image (multiview stays mono) and no shader is patched. Driveclub read them from its eboot.
bool OutputSize(u32& width, u32& height) {
    width = height = 0;
    return false;
}

bool RenderSize(u32& width, u32& height) {
    width = height = 0;
    return false;
}

void PatchShader(u64 hash, std::vector<u32>& spirv) {
    (void)hash;
    (void)spirv;
}

} // namespace Vulkan::Vr::Fixes
