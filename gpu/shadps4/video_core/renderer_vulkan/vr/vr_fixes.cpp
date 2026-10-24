// SPDX-License-Identifier: GPL-2.0-or-later
// VR (dcport-vr): see vr_fixes.h.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
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

// Uncharted fork: the collection renders at 1920x1080, at the eye size run.py patched in (BB_VR_RENDER_SIZE,
// scripts/vr_resolution.py), or flat at the community build's BB_GAME_RES (src/runtime_display.c; even sizes from
// 640x360 as it reads them). Driveclub read its sizes from its eboot.
bool RenderSize(u32& width, u32& height) {
    static const std::pair<u32, u32> size = [] {
        u32 w = 0, h = 0;
        if (const char* res = std::getenv("BB_VR_RENDER_SIZE"); res && std::sscanf(res, "%ux%u", &w, &h) == 2) {
            return std::pair{w, h};
        }
        if (const char* res = std::getenv("BB_GAME_RES");
            res && std::sscanf(res, "%ux%u", &w, &h) == 2 && w >= 640 && h >= 360) {
            return std::pair{w & ~1u, h & ~1u};
        }
        return std::pair{1920u, 1080u};
    }();
    width = size.first;
    height = size.second;
    return true;
}

bool OutputSize(u32& width, u32& height) {
    return RenderSize(width, height);
}

bool MovieCanvas(s32& width, s32& height) {
    u32 w = 0, h = 0;
    const char* res = std::getenv("BB_VR_RENDER_SIZE");
    if (!res || std::sscanf(res, "%ux%u", &w, &h) != 2 || !w || !h) {
        return false;
    }
    // The games' movie buffers hold a 1080p frame (1920 x 1088 NV12; they do not ask the decoder for more): the largest
    // canvas of the render's shape in as many pixels, rows a multiple of 64 and the pitch of 16 (the decoder's copy).
    height = s32(std::sqrt(1920.0 * 1088.0 * h / w)) & ~63;
    width = s32(u64(height) * w / h) & ~15;
    return true;
}

void PatchShader(u64 hash, std::vector<u32>& spirv) {
    (void)hash;
    (void)spirv;
}

namespace {
// The latest movie frame's layout and its buffer's address (a game decodes into a few buffers in turn: any of them).
std::atomic<u32> g_movie_pitch{0}, g_movie_width{0}, g_movie_height{0}, g_movie_rows{0};
std::array<std::atomic<u64>, 8> g_movie_buffers{};
std::atomic<u32> g_movie_next{0};
std::atomic<s64> g_movie_drawn_ns{0}; // the movie shader's last dispatch (steady clock)

bool MovieBuffer(u64 address) {
    for (const auto& buffer : g_movie_buffers) {
        const u64 b = buffer.load(std::memory_order_relaxed);
        if (b && address >= b && address < b + 0x1000) { // the luma may start past a header
            return true;
        }
    }
    return false;
}

// Writes `image` back where the shader's descriptor comes from (its flat user data): false when a dword the fix
// changes (base, size, pitch: dwords 0-4) is an immediate in the shader.
bool Store(const Shader::Info& info, const Shader::ImageResource& res, const AmdGpu::Image& image) {
    const auto& fetch = res.sharp_fetch;
    u32 dwords[sizeof(AmdGpu::Image) / 4];
    std::memcpy(dwords, &image, sizeof(dwords));
    const u32 used = res.is_r128 ? 4 : u32(std::size(dwords));
    for (u32 i = 0; i < used && i < 5; ++i) {
        if (!((fetch.load_mask >> i) & 1)) {
            return false;
        }
    }
    u32* flat = const_cast<u32*>(info.FlatUserData().data());
    for (u32 i = 0; i < used; ++i) {
        if ((fetch.load_mask >> i) & 1) {
            flat[fetch.offsets[i]] = dwords[i];
        }
    }
    return true;
}
} // namespace

void NoteMovieFrame(u64 buffer, u32 pitch, u32 width, u32 height, u32 rows) {
    g_movie_pitch.store(pitch, std::memory_order_relaxed);
    g_movie_width.store(width, std::memory_order_relaxed);
    g_movie_height.store(height, std::memory_order_relaxed);
    g_movie_rows.store(rows, std::memory_order_relaxed);
    for (const auto& b : g_movie_buffers) {
        if (b.load(std::memory_order_relaxed) == buffer) {
            return;
        }
    }
    g_movie_buffers[g_movie_next.fetch_add(1) % g_movie_buffers.size()].store(buffer, std::memory_order_relaxed);
}

void FixMovieImages(const Shader::Info& cs) {
    const u32 pitch = g_movie_pitch.load(std::memory_order_relaxed);
    u32 rw = 0, rh = 0;
    if (!pitch || cs.images.size() < 2 || !RenderSize(rw, rh)) {
        return;
    }
    AmdGpu::Image luma = cs.images[0].GetSharp(cs);
    AmdGpu::Image chroma = cs.images[1].GetSharp(cs);
    if (luma.GetDataFmt() != AmdGpu::DataFormat::Format8 || chroma.GetDataFmt() != AmdGpu::DataFormat::Format8_8 ||
        !MovieBuffer(luma.Address()) ||
        chroma.Address() != luma.Address() + u64(u32(luma.pitch) + 1) * (u32(luma.height) + 1)) {
        return;
    }
    g_movie_drawn_ns.store(std::chrono::steady_clock::now().time_since_epoch().count(), std::memory_order_relaxed);
    const u32 width = g_movie_width.load(std::memory_order_relaxed);
    const u32 height = g_movie_height.load(std::memory_order_relaxed);
    const u32 rows = g_movie_rows.load(std::memory_order_relaxed);
    if (u32(luma.width) + 1 == width && u32(luma.height) + 1 == height && u32(luma.pitch) + 1 == pitch) {
        return; // the game's own size: its images are right
    }
    luma.width = width - 1;
    luma.height = height - 1;
    luma.pitch = pitch - 1;
    chroma.width = width / 2 - 1;
    chroma.height = height / 2 - 1;
    chroma.pitch = pitch / 2 - 1;
    chroma.base_address = (luma.Address() + u64(pitch) * rows) >> 8;
    const bool done = Store(cs, cs.images[0], luma) && Store(cs, cs.images[1], chroma);
    static std::atomic<bool> told{false};
    if (!told.exchange(true)) {
        std::printf("VR fixes: movie shader %#llx: luma and chroma %s %ux%u (render size %ux%u)\n",
                    (unsigned long long)cs.pgm_hash, done ? "made" : "could not be made", width, height, rw, rh);
        std::fflush(stdout);
    }
}

bool MovieRect(u32& width, u32& height) {
    const s64 drawn = g_movie_drawn_ns.load(std::memory_order_relaxed);
    if (!drawn || std::chrono::steady_clock::now().time_since_epoch().count() - drawn > 300'000'000) {
        return false;
    }
    width = g_movie_width.load(std::memory_order_relaxed);
    height = g_movie_height.load(std::memory_order_relaxed);
    return width && height;
}

} // namespace Vulkan::Vr::Fixes
