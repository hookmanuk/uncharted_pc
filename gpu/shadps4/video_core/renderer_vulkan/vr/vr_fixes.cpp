// SPDX-License-Identifier: GPL-2.0-or-later
// VR (dcport-vr): see vr_fixes.h.

#include <cstdio>
#include <cstdlib>
#include <utility>

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

// Uncharted fork: the collection renders at 1920x1080 and shows that (BB_RENDER_RES: the size the port patches in at
// start). Driveclub read its sizes from its eboot.
bool RenderSize(u32& width, u32& height) {
    static const std::pair<u32, u32> size = [] {
        u32 w = 1920, h = 1080;
        if (const char* res = std::getenv("BB_RENDER_RES"); res && std::sscanf(res, "%ux%u", &w, &h) != 2) {
            w = 1920, h = 1080;
        }
        return std::pair{w, h};
    }();
    width = size.first;
    height = size.second;
    return true;
}

bool OutputSize(u32& width, u32& height) {
    return RenderSize(width, height);
}

void PatchShader(u64 hash, std::vector<u32>& spirv) {
    (void)hash;
    (void)spirv;
}

} // namespace Vulkan::Vr::Fixes
