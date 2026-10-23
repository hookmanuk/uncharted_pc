// SPDX-License-Identifier: GPL-2.0-or-later
// VR (dcport-vr): Driveclub's camera hooks. See vr_camera.h.
//
// The setters copy the world matrix (rsi) and the projection (rdx) into the view (rdi). Their first 21 bytes are three
// position-independent stores; they are replaced by "mov rax, stub; jmp rax". The stub saves the caller-saved
// registers, calls Handler (Windows ABI) with the saved block, which may point rsi and rdx at modified copies, then
// runs the displaced 21 bytes and jumps back.
//
// Row-vector matrices, as the game keeps them: v_world = v_local * W (rows: right, up, back, position; the camera
// looks down -Z, Y up: OpenXR's convention), clip = v_view * P (P[2][3] = -1, reversed infinite depth: P[3][2] = near).
//
// BB_VR_HEAD=0: hooks on, matrices untouched. BB_VR_CAMLOG=1: each view object's projection and position, once.

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include <xbyak/xbyak.h>

#include "video_core/renderer_vulkan/vr/vr_camera.h"
#include "video_core/renderer_vulkan/vr/vr_multiview.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace Vulkan::Vr::Camera {
namespace {

u64 g_image_base = 0; // the game image (the loader maps it where it finds room: found by the first site's bytes)

extern "C" void SceneHandler(u64* regs);
extern "C" void ViewHandler(u64* regs);
extern "C" void CameraSetHandler(u64* regs);
extern "C" void ProjectionHandler(u64* regs);

struct Site {
    u64 offset;
    std::vector<u8> bytes; // the displaced instructions (position-independent), at least 12 bytes
    void (*handler)(u64*);
};

// The game's camera setters to hook. Driveclub's were Camera::Set (+0x72d9e0, CameraSetHandler), Scene::SetCamera
// (+0x5e7a20, SceneHandler) and its two view setters (+0x736650, +0x7376e0, ViewHandler): see the Driveclub recomp's
// vr branch.
//
// Uncharted 1 (eboot.bin, CUSA02343 01.00): a view's projection update (image +0x2c8dd0, rdi = the view). It builds the
// projection from the view's aspect (+0x290, the field of view's height / width), x scale (+0x294), near (+0x298) and
// far (+0x29c), then (+0x2c9070) every matrix of the view from its camera-to-world input: three rows at +0x250 (the
// camera's x: left, y: up, z: forward) and the position at +0x280 (row vectors, metres). The render camera object is
// the image's +0xa86b30 (position, orientation, field of view in degrees at +0x20); its view is at +0x30. The camera
// update (+0xa6f.. +0xa7178) writes the input and the field of view every frame, then calls this. The outputs: +0x10
// camera to world, +0x50 world to camera, +0x90 projection (P00 negative: x mirrored), +0xd0 view-projection.
constexpr u64 U1MainView = 0xa86b60;
const std::vector<Site> Sites = {
    {0x2c8dd0, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x48, 0x81, 0xec, 0xa0, 0x00, 0x00, 0x00}, &ProjectionHandler},
};

struct alignas(16) Mat {
    float m[16];
};

// The saved register block (see Stub): lowest address first.
enum Reg { R11, R10, R9, R8, RDI, RSI, RDX, RCX, RAX, RegCount };

struct Head {
    float pose[16];
    float tan[4]; // left (negative), right, up, down (negative)
    float layer[7];
};

std::mutex g_head_mutex;
Head g_head{};
bool g_head_valid = false;
Head g_rendered{};
bool g_rendered_valid = false;
// The last player views' rendered positions with their heads (RenderedAt): the GPU renders a frame or two behind.
struct Rendered {
    float position[3];
    Head head;
};
constexpr u32 HistorySize = 32;
Rendered g_history[HistorySize];
u32 g_history_count = 0;

// Copies handed to the setters: a ring, as the setters may run on more than one thread and the game copies them at once.
constexpr u32 RingSize = 64;
struct alignas(16) CameraDesc {
    u8 bytes[0x1000]; // the description runs past +0x2d0 (nested view settings); copied whole
};
CameraDesc g_ring[RingSize];
std::atomic<u32> g_ring_next{0};

bool g_head_enabled = true;
bool g_camlog = false;
int g_fov_mode = 2; // BB_VR_FOV: 0 the game's projection, 1 symmetric headset field of view, 2 the headset's (asymmetric)
std::atomic<int> g_log_next{0};
std::atomic<float> g_near{0.f};

void Multiply(const float* a, const float* b, float* out) {
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            out[r * 4 + c] = a[r * 4 + 0] * b[0 * 4 + c] + a[r * 4 + 1] * b[1 * 4 + c] + a[r * 4 + 2] * b[2 * 4 + c] +
                             a[r * 4 + 3] * b[3 * 4 + c];
        }
    }
}

// The player's view: a perspective projection of the screen's aspect, wider than the mirrors' (P00 below 2: 0.97 at
// 16:9, 1.725 square; the mirror 2.53).
bool IsPlayerView(const float* proj) {
    const float sx = proj[0], sy = proj[5];
    const float ratio = sy / sx / ScreenAspect();
    return sx > 0.3f && sx < 2.f && ratio > 0.96f && ratio < 1.045f && proj[11] == -1.f;
}

void LogView(u64 view, const float* world, const float* proj) {
    static std::mutex mutex;
    static std::vector<u64> seen;
    std::scoped_lock lock{mutex};
    if (std::find(seen.begin(), seen.end(), view) != seen.end() || seen.size() > 64) {
        return;
    }
    seen.push_back(view);
    std::printf("VR camera: view %#llx proj %.4g %.4g %.4g %.4g | %.4g %.4g pos %.5g %.5g %.5g%s\n",
                (unsigned long long)view, proj[0], proj[5], proj[10], proj[14], proj[8], proj[9], world[12], world[13],
                world[14], IsPlayerView(proj) ? " (player)" : "");
    std::fflush(stdout);
}

extern "C" void ViewHandler(u64* regs) {
    const auto* world = reinterpret_cast<const float*>(regs[RSI]);
    const auto* proj = reinterpret_cast<const float*>(regs[RDX]);
    if (g_camlog) {
        LogView(regs[RDI], world, proj);
    }
    if (g_log_next.load(std::memory_order_relaxed) > 0 && g_log_next.fetch_sub(1) > 0) {
        std::printf("VR camera: call view %#llx proj %.4g %.4g %.4g %.4g | %.4g %.4g pos %.5g %.5g %.5g fwd %.3g %.3g %.3g%s\n",
                    (unsigned long long)regs[RDI], proj[0], proj[5], proj[10], proj[14], proj[8], proj[9], world[12],
                    world[13], world[14], -world[8], -world[9], -world[10], IsPlayerView(proj) ? " (player)" : "");
        std::fflush(stdout);
    }
}

// The game culls with the frustum of the camera it sets here (Camera::Set(camera, world, block): the block holds the
// projection, then at +0x60 six view-space planes, outward normals, inside where n.p + d <= 0: near, left, right,
// bottom, top, far). Its own camera's planes face forward, +-46 degrees across: as the head turned, parts of the car
// beside the driver and of cars alongside were culled (Matt 2026-10-19: the passenger door). The player camera's
// planes are the headset's field of view turned with the head instead; the world and the projection stay the game's
// (the detail it picks is unchanged), and the block is a copy (the game's is rebuilt every frame).
// BB_VR_CULL_HEAD=0 turns it off.
alignas(16) float g_cull_block[0x110 / 4];

void HeadPlane(const float* pose, float nx, float ny, float nz, float d, float* out) {
    const float inv = 1.f / std::sqrt(nx * nx + ny * ny + nz * nz);
    nx *= inv, ny *= inv, nz *= inv, d *= inv;
    // Row vectors: a head direction v goes to v * R (the pose's upper 3x3). The planes stay at the eye: the head's
    // position moves them by centimetres in the car (the margin covers it), and in the menus it is metres away.
    const float cx = nx * pose[0] + ny * pose[4] + nz * pose[8];
    const float cy = nx * pose[1] + ny * pose[5] + nz * pose[9];
    const float cz = nx * pose[2] + ny * pose[6] + nz * pose[10];
    out[0] = cx, out[1] = cy, out[2] = cz;
    out[3] = d;
}

extern "C" void CameraSetHandler(u64* regs) {
    static const bool enabled = !(std::getenv("BB_VR_CULL_HEAD") && std::getenv("BB_VR_CULL_HEAD")[0] == '0');
    const auto* block = reinterpret_cast<const float*>(regs[RDX]);
    if (!enabled || !g_head_enabled || !IsPlayerView(block)) {
        return;
    }
    Head head;
    {
        std::scoped_lock lock{g_head_mutex};
        if (!g_head_valid) {
            return;
        }
        head = g_head;
    }
    std::memcpy(g_cull_block, block, sizeof(g_cull_block));
    // Both eyes: the headset's field of view as rendered (the multiview overscan), and a little more for the eyes'
    // offsets.
    const float margin = (Multiview::Enabled() ? 0.08f : 0.f) + 0.05f;
    const float l = head.tan[0] - margin, r = head.tan[1] + margin, u = head.tan[2] + margin, d = head.tan[3] - margin;
    float* planes = g_cull_block + 0x60 / 4;
    const float near_d = planes[3], far_d = planes[23];
    HeadPlane(head.pose, 0.f, 0.f, 1.f, near_d, planes + 0);
    HeadPlane(head.pose, -1.f, 0.f, -l, 0.f, planes + 4);
    HeadPlane(head.pose, 1.f, 0.f, r, 0.f, planes + 8);
    HeadPlane(head.pose, 0.f, -1.f, -d, 0.f, planes + 12);
    HeadPlane(head.pose, 0.f, 1.f, u, 0.f, planes + 16);
    HeadPlane(head.pose, 0.f, 0.f, -1.f, far_d, planes + 20);
    static bool told = false;
    if (!told) {
        told = true;
        std::printf("VR camera: culling with the head (the game's left plane %.4f %.4f %.4f %.4g)\n", block[28],
                    block[29], block[30], block[31]);
        std::fflush(stdout);
    }
    regs[RDX] = reinterpret_cast<u64>(g_cull_block);
}

extern "C" void SceneHandler(u64* regs) {
    const auto* desc = reinterpret_cast<const float*>(regs[RSI]);
    const float* world = desc;
    const float* proj = desc + 16;
    if (g_log_next.load(std::memory_order_relaxed) > 0) {
        std::printf("VR camera: scene %#llx desc %#llx rdx %#llx rcx %#llx ret %#llx proj %.4g %.4g pos %.5g %.5g %.5g\n",
                    (unsigned long long)regs[RDI], (unsigned long long)regs[RSI], (unsigned long long)regs[RDX],
                    (unsigned long long)regs[RCX], (unsigned long long)regs[RegCount], proj[0], proj[5], world[12],
                    world[13], world[14]);
        std::fflush(stdout);
    }
    if (!g_head_enabled || !IsPlayerView(proj)) {
        return;
    }
    // Development (BB_VR_LOOKLOG=1): the game's player camera yaw (its forward on the ground plane, degrees) per call.
    static const bool look_log = std::getenv("BB_VR_LOOKLOG") != nullptr;
    if (look_log) {
        static u64 calls = 0;
        std::printf("VR look: %llu yaw %.4f fwd %.5f %.5f %.5f proj %.4f %.4f\n", (unsigned long long)calls++,
                    std::atan2(-world[8], -world[10]) * 57.29578f, -world[8], -world[9], -world[10], proj[0], proj[5]);
        std::fflush(stdout);
    }
    Head head;
    {
        std::scoped_lock lock{g_head_mutex};
        if (!g_head_valid) {
            return;
        }
        head = g_head;
        g_rendered = g_head;
        g_rendered_valid = true;
    }
    g_near = proj[14];
    CameraDesc& copy = g_ring[g_ring_next.fetch_add(1) % RingSize];
    std::memcpy(copy.bytes, desc, sizeof(copy.bytes));
    auto* new_world = reinterpret_cast<float*>(copy.bytes);
    auto* new_proj = new_world + 16;
    Multiply(head.pose, world, new_world);

    // The headset's field of view, the game's depth terms kept (reversed infinite: P[3][2] = near).
    float l = head.tan[0], r = head.tan[1], u = head.tan[2], d = head.tan[3];
    if (g_fov_mode == 1) {
        const float x = std::max(-l, r), y = std::max(u, -d);
        l = -x, r = x, u = y, d = -y;
    }
    if (Multiview::Enabled()) {
        // Each eye is this view shifted sideways: near surfaces move inwards, leaving the outer edge of each eye
        // unrendered. Rendering a little wider (the layer says so) puts that edge outside what the headset shows.
        static const float overscan = [] {
            const char* v = std::getenv("BB_VR_OVERSCAN");
            return v ? float(std::atof(v)) : 0.08f;
        }();
        l -= overscan;
        r += overscan;
    }
    if (g_fov_mode != 0) {
        new_proj[0] = 2.f / (r - l);
        new_proj[5] = 2.f / (u - d);
        new_proj[8] = (r + l) / (r - l);
        new_proj[9] = (u + d) / (u - d);
    }
    {
        std::scoped_lock lock{g_head_mutex};
        // What the layer must say: the field of view actually rendered.
        if (g_fov_mode == 0) {
            g_rendered.tan[0] = (-1.f - new_proj[8]) / new_proj[0];
            g_rendered.tan[1] = (1.f - new_proj[8]) / new_proj[0];
            g_rendered.tan[2] = (1.f - new_proj[9]) / new_proj[5];
            g_rendered.tan[3] = (-1.f - new_proj[9]) / new_proj[5];
        } else {
            g_rendered.tan[0] = l, g_rendered.tan[1] = r, g_rendered.tan[2] = u, g_rendered.tan[3] = d;
        }
        Rendered& entry = g_history[g_history_count++ % HistorySize];
        std::memcpy(entry.position, new_world + 12, sizeof(entry.position));
        entry.head = g_rendered;
    }
    regs[RSI] = reinterpret_cast<u64>(copy.bytes);
}

// Uncharted 1: the main camera's view (rdi) gets the head. Its camera-to-world input becomes head x camera (the head's
// pose turned into the game camera's axes, x left and z forward: C = diag(-1, 1, -1) on both sides), and its field of
// view the headset's (symmetric: the game builds a symmetric projection; the layer says what was rendered). The game
// rebuilds the input every frame; should it not, its own is put back before the head goes on again.
alignas(16) float g_input_original[16];
alignas(16) float g_input_written[16];
bool g_input_written_valid = false;

extern "C" void ProjectionHandler(u64* regs) {
    auto* view = reinterpret_cast<u8*>(regs[RDI]);
    if (u64(view) != g_image_base + U1MainView) {
        return;
    }
    auto* input = reinterpret_cast<float*>(view + 0x250);  // rows x, y, z, position (w unused)
    auto* params = reinterpret_cast<float*>(view + 0x290); // aspect, x scale, near, far
    g_near = params[2];
    if (g_log_next.load(std::memory_order_relaxed) > 0 && g_log_next.fetch_sub(1) > 0) {
        std::printf("VR camera: view %#llx pos %.3f %.3f %.3f fwd %.3f %.3f %.3f aspect %.4f xscale %.4f near %.4f "
                    "far %.1f\n",
                    (unsigned long long)regs[RDI], input[12], input[13], input[14], input[8], input[9], input[10],
                    params[0], params[1], params[2], params[3]);
        std::fflush(stdout);
    }
    if (!g_head_enabled) {
        return;
    }
    Head head;
    {
        std::scoped_lock lock{g_head_mutex};
        if (!g_head_valid) {
            return;
        }
        head = g_head;
        g_rendered = g_head;
        g_rendered_valid = true;
    }
    if (g_input_written_valid && std::memcmp(input, g_input_written, sizeof(g_input_written)) == 0) {
        std::memcpy(input, g_input_original, sizeof(g_input_original));
    }
    std::memcpy(g_input_original, input, sizeof(g_input_original));
    static const float world_scale = [] {
        const char* v = std::getenv("BB_VR_WORLD_SCALE");
        return v ? float(std::atof(v)) : 1.f;
    }();
    constexpr float axes[4] = {-1.f, 1.f, -1.f, 1.f};
    float h[16], in[16], out[16];
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            h[r * 4 + c] = head.pose[r * 4 + c] * axes[r] * axes[c];
        }
    }
    h[12] *= world_scale, h[13] *= world_scale, h[14] *= world_scale;
    std::memcpy(in, input, sizeof(in));
    in[3] = in[7] = in[11] = 0.f;
    in[15] = 1.f;
    Multiply(h, in, out);
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 3; ++c) {
            input[r * 4 + c] = out[r * 4 + c];
        }
    }
    std::memcpy(g_input_written, input, sizeof(g_input_written));
    g_input_written_valid = true;

    float l = head.tan[0], r = head.tan[1], u = head.tan[2], d = head.tan[3];
    if (Multiview::Enabled()) { // the eyes' sideways shift uncovers each eye's outer edge (see SceneHandler)
        static const float overscan = [] {
            const char* v = std::getenv("BB_VR_OVERSCAN");
            return v ? float(std::atof(v)) : 0.08f;
        }();
        l -= overscan;
        r += overscan;
    }
    const float tx = std::max(-l, r), ty = std::max(u, -d);
    params[0] = ty / tx;
    params[1] = 1.f / tx;
    std::scoped_lock lock{g_head_mutex};
    g_rendered.tan[0] = -tx, g_rendered.tan[1] = tx, g_rendered.tan[2] = ty, g_rendered.tan[3] = -ty;
    Rendered& entry = g_history[g_history_count++ % HistorySize];
    std::memcpy(entry.position, out + 12, sizeof(entry.position));
    entry.head = g_rendered;
}

struct Stub : Xbyak::CodeGenerator {
    Stub(const Site& site) : Xbyak::CodeGenerator(4096) {
        using namespace Xbyak::util;
        push(rax);
        push(rcx);
        push(rdx);
        push(rsi);
        push(rdi);
        push(r8);
        push(r9);
        push(r10);
        push(r11);
        // Entry rsp = 8 mod 16; 9 pushes make it 16-aligned. xmm0-7 (argument registers) and the shadow space.
        sub(rsp, 0xa0);
        for (int i = 0; i < 8; ++i) {
            movdqu(ptr[rsp + 0x20 + i * 16], Xbyak::Xmm(i));
        }
        lea(rcx, ptr[rsp + 0xa0]);
        mov(rax, reinterpret_cast<u64>(site.handler));
        call(rax);
        for (int i = 0; i < 8; ++i) {
            movdqu(Xbyak::Xmm(i), ptr[rsp + 0x20 + i * 16]);
        }
        add(rsp, 0xa0);
        pop(r11);
        pop(r10);
        pop(r9);
        pop(r8);
        pop(rdi);
        pop(rsi);
        pop(rdx);
        pop(rcx);
        pop(rax);
        db(site.bytes.data(), site.bytes.size());
        jmp(ptr[rip]);
        dq(g_image_base + site.offset + site.bytes.size());
    }
};

std::once_flag g_install_once;
std::vector<Stub*> g_stubs;

void DoInstall() {
#ifdef _WIN32
    if (const char* v = std::getenv("BB_VR"); !v || v[0] != '1') { // opt-in, see Vr::Init
        return;
    }
    g_head_enabled = !(std::getenv("BB_VR_HEAD") && std::getenv("BB_VR_HEAD")[0] == '0');
    g_camlog = std::getenv("BB_VR_CAMLOG") != nullptr;
    if (const char* v = std::getenv("BB_VR_FOV")) {
        g_fov_mode = std::atoi(v);
    }
    for (const Site& site : Sites) {
        auto* code = reinterpret_cast<u8*>(g_image_base + site.offset);
        MEMORY_BASIC_INFORMATION mbi{};
        if (!VirtualQuery(code, &mbi, sizeof(mbi)) || mbi.State != MEM_COMMIT ||
            std::memcmp(code, site.bytes.data(), site.bytes.size()) != 0) {
            std::printf("VR camera: the view setter at +%#llx is not as expected: camera hooks off\n",
                        (unsigned long long)site.offset);
            std::fflush(stdout);
            return;
        }
    }
    const bool cull_head = !(std::getenv("BB_VR_CULL_HEAD") && std::getenv("BB_VR_CULL_HEAD")[0] == '0');
    for (const Site& site : Sites) {
        if (site.handler == &CameraSetHandler && !cull_head) {
            continue;
        }
        auto* stub = new Stub(site);
        stub->ready();
        g_stubs.push_back(stub);
        auto* code = reinterpret_cast<u8*>(g_image_base + site.offset);
        std::vector<u8> patch(site.bytes.size(), 0x90);
        patch[0] = 0x48; // mov rax, imm64
        patch[1] = 0xb8;
        const u64 target = reinterpret_cast<u64>(stub->getCode());
        std::memcpy(patch.data() + 2, &target, 8);
        patch[10] = 0xff; // jmp rax
        patch[11] = 0xe0;
        DWORD old = 0;
        VirtualProtect(code, patch.size(), PAGE_EXECUTE_READWRITE, &old);
        std::memcpy(code, patch.data(), patch.size());
        VirtualProtect(code, patch.size(), old, &old);
        FlushInstructionCache(GetCurrentProcess(), code, patch.size());
    }
    std::printf("VR camera: view setters hooked%s\n", g_head_enabled ? "" : " (BB_VR_HEAD=0: matrices untouched)");
    std::fflush(stdout);
#endif
}

} // namespace

std::atomic<bool> g_image_mapped{false}; // the game's image is mapped (Install saw it)

float ScreenAspect() {
    // The HUD's canvas. Uncharted fork: 16:9 until each game's screen aspect is found (Driveclub read its eboot's).
    return 16.f / 9.f;
}

void Install() {
#ifdef _WIN32
    if (Sites.empty()) { // no camera hooks for this game yet: the game on a flat screen
        static std::once_flag once;
        std::call_once(once, [] {
            std::printf("VR camera: no camera hooks for this game yet (the game on a flat screen)\n");
            std::fflush(stdout);
        });
        return;
    }
    // The game image is mapped before the first frame: the executable allocation holding the first site's bytes.
    if (!g_image_base) {
        static int calls = 0;
        if (calls++ % 60 != 0) {
            return;
        }
        const Site& site = Sites[0];
        MEMORY_BASIC_INFORMATION mbi{};
        for (u64 at = 0x400000; at < 0x80000000ull && VirtualQuery(reinterpret_cast<void*>(at), &mbi, sizeof(mbi));
             at = u64(mbi.BaseAddress) + mbi.RegionSize) {
            const u64 base = u64(mbi.BaseAddress);
            if (mbi.State == MEM_COMMIT && u64(mbi.AllocationBase) == base &&
                (mbi.Protect & (PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE)) &&
                site.offset + site.bytes.size() <= mbi.RegionSize &&
                std::memcmp(reinterpret_cast<const void*>(base + site.offset), site.bytes.data(), site.bytes.size()) ==
                    0) {
                g_image_base = base;
                std::printf("VR camera: game image at %#llx\n", (unsigned long long)base);
                std::fflush(stdout);
                break;
            }
        }
        if (!g_image_base) {
            return;
        }
    }
#endif
    g_image_mapped.store(true, std::memory_order_release);
    std::call_once(g_install_once, DoInstall);
}

float Near() {
    return g_near.load();
}

void LogNext(int count) {
    g_log_next = count;
}

void SetHead(const float (&pose)[16], float tan_left, float tan_right, float tan_up, float tan_down,
             const float (&layer_pose)[7]) {
    std::scoped_lock lock{g_head_mutex};
    std::memcpy(g_head.pose, pose, sizeof(g_head.pose));
    std::memcpy(g_head.layer, layer_pose, sizeof(g_head.layer));
    g_head.tan[0] = tan_left;
    g_head.tan[1] = tan_right;
    g_head.tan[2] = tan_up;
    g_head.tan[3] = tan_down;
    g_head_valid = true;
}

bool LastRendered(float (&layer_pose)[7], float (&tangents)[4]) {
    std::scoped_lock lock{g_head_mutex};
    if (!g_rendered_valid) {
        return false;
    }
    std::memcpy(layer_pose, g_rendered.layer, sizeof(layer_pose));
    std::memcpy(tangents, g_rendered.tan, sizeof(tangents));
    return true;
}

u32 RecentPositions(float (*positions)[3], u32 max) {
    std::scoped_lock lock{g_head_mutex};
    const u32 count = std::min({g_history_count, HistorySize, max});
    for (u32 i = 0; i < count; ++i) {
        std::memcpy(positions[i], g_history[(g_history_count - count + i) % HistorySize].position, sizeof(positions[i]));
    }
    return count;
}

bool RenderedAt(const float* position, float (&layer_pose)[7], float (&tangents)[4], float& distance) {
    std::scoped_lock lock{g_head_mutex};
    if (std::abs(position[0]) + std::abs(position[1]) + std::abs(position[2]) < 1e-3f) {
        return false; // the origin: loading screens' cameras, or not a camera at all
    }
    const u32 count = std::min(g_history_count, HistorySize);
    const Rendered* best = nullptr;
    float best_d2 = 0.f;
    for (u32 n = 0; n < count; ++n) { // oldest to newest: the newest of equal matches wins
        const Rendered& e = g_history[(g_history_count - count + n) % HistorySize];
        const float dx = e.position[0] - position[0], dy = e.position[1] - position[1], dz = e.position[2] - position[2];
        const float d2 = dx * dx + dy * dy + dz * dz;
        if (!best || d2 <= best_d2) {
            best = &e;
            best_d2 = d2;
        }
    }
    if (!best) {
        return false;
    }
    distance = std::sqrt(best_d2);
    std::memcpy(layer_pose, best->head.layer, sizeof(layer_pose));
    std::memcpy(tangents, best->head.tan, sizeof(tangents));
    return true;
}

} // namespace Vulkan::Vr::Camera
