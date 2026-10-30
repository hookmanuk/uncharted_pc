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
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include <xbyak/xbyak.h>

#include "video_core/renderer_vulkan/vr/vr_camera.h"
#include "video_core/renderer_vulkan/vr/vr_fixes.h"
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
extern "C" void AimHandler(u64* regs);

// A byte pattern with wildcards ("??": the operands that move with the code, call and jump targets and rip-relative
// displacements), so a game version that only shifts its code is found all the same.
struct Sig {
    std::vector<int> bytes; // -1: any
    explicit Sig(const char* text) {
        for (const char* at = text; *at;) {
            if (*at == ' ') {
                ++at;
            } else if (*at == '?') {
                bytes.push_back(-1);
                at += 2;
            } else {
                bytes.push_back(int(std::strtoul(std::string(at, 2).c_str(), nullptr, 16)));
                at += 2;
            }
        }
    }
    bool At(const u8* code) const {
        for (size_t i = 0; i < bytes.size(); ++i) {
            if (bytes[i] >= 0 && code[i] != u8(bytes[i])) {
                return false;
            }
        }
        return true;
    }
};

// Every place in [code, code + size) where the pattern matches (at most `limit`).
std::vector<u64> Find(const u8* code, u64 size, const Sig& sig, size_t limit = 16) {
    std::vector<u64> out;
    for (u64 i = 0; i + sig.bytes.size() <= size && out.size() < limit; ++i) {
        if ((sig.bytes[0] < 0 || code[i] == u8(sig.bytes[0])) && sig.At(code + i)) {
            out.push_back(i);
        }
    }
    return out;
}

struct Site {
    u64 address = 0; // found by the game's pattern (Locate)
    std::vector<u8> bytes; // the displaced instructions (position-independent), read from the game: at least 12 bytes
    void (*handler)(u64*) = nullptr;
};

// Uncharted 1 (1.00 offsets, for reading): a view's projection update (image +0x2c8dd0, rdi = the view). It builds the
// projection from the view's aspect (+0x290, the field of view's height / width), x scale (+0x294), near (+0x298) and
// far (+0x29c), then every matrix of the view from its camera-to-world input: three rows at +0x250 (the camera's x:
// left, y: up, z: forward) and the position at +0x280 (row vectors, metres). The camera update (+0xa6f.. +0xa7178)
// writes the input and the field of view every frame, then calls this. The outputs: +0x10 camera to world, +0x50 world
// to camera, +0x90 projection (P00 negative: x mirrored), +0xd0 view-projection.
//
// Uncharted 2 (big2-ps4_Shipping.elf): the same view, its fields 0x40 lower (input +0x210, position +0x240, aspect,
// x scale, near, far at +0x250), built in one function (image +0x3e2c50). Each frame's render views are copies in GPU
// memory, built from the call at +0x61194a (also the shadow cascades: x scale 0, near 2.5 and more); the static view in
// the image (+0xeb81e0) is a wide (100 degree) one, not the one drawn. The shaders get the near plane as it is (U1's
// get half of it). Uncharted 3: the same, +0x80a2f0 and the call at +0x5b153a.
//
// The game's camera is found by its code, not by addresses, so every version of a game is the same entry: the
// projection update by its opening instructions (func) and the camera update's call to it (caller: the main view's
// return address), both masked where the code is relocated.
struct Game {
    const char* name;
    const char* func;   // the projection update (identifies the game: one match in the code)
    u32 displaced;      // bytes of whole instructions from its start that the hook replaces (12 or more)
    const char* caller; // the main camera's call to it, ending with the call (e8 ?? ?? ?? ??)
    u32 input;          // the camera-to-world input: three rows and the position (row vectors)
    u32 params;         // aspect, x scale, near, far
    float shader_near;  // the near plane in the shaders' constants, per unit of the view's
    // Head aim: the aiming camera's update (rsi: the camera object), whose aim direction (x, y, z: the camera's
    // forward) at `aim_direction` it turns by the right stick, keeps from frame to frame and the player's aim reads.
    // Its opening, `aim_displaced` bytes of whole instructions replaced. nullptr: no head aim.
    const char* aim = nullptr;
    u32 aim_displaced = 0;
    u32 aim_direction = 0;
};
// (tools/vr/sigmake.py makes the patterns from a plaintext ELF.)
const std::vector<Game> Games = {
    {"Uncharted 1",
     "55 48 89 e5 41 56 53 48 81 ec a0 00 00 00 48 89 fb 4c 8b 35 ?? ?? ?? ?? 49 8b 06 48 89 45 e8 80 bb a4 02 00 00 00",
     14, "c5 f2 5e c0 c5 fa 11 83 c4 02 00 00 4c 89 ff e8 ?? ?? ?? ??", 0x250, 0x290, 0.5f,
     // (1.00: +0xc42c0; the direction at +0x1830 found by turning a copy of each candidate, as RPCS3's was)
     "55 48 89 e5 41 57 41 56 41 55 41 54 53 48 81 ec 68 02 00 00 49 89 d7 48 89 f3 49 89 fe", 13, 0x1830},
    {"Uncharted 2",
     "55 48 89 e5 53 48 81 ec 28 01 00 00 48 89 fb 80 bb 82 02 00 00 00 0f 85 ?? ?? ?? ??", 12,
     "49 89 cf 49 89 d4 48 89 f3 49 89 fe 48 89 df e8 ?? ?? ?? ??", 0x210, 0x250, 1.0f},
    // Uncharted 3 (big3-ps4_Shipping.elf): U2's view and builder, the same code.
    {"Uncharted 3",
     "55 48 89 e5 53 48 81 ec 08 01 00 00 48 89 fb 80 bb 6c 02 00 00 00 0f 85 ?? ?? ?? ??", 12,
     "49 89 cf 49 89 d4 48 89 f3 49 89 fe 48 89 df e8 ?? ?? ?? ??", 0x210, 0x250, 1.0f},
};
const Game* g_game = nullptr;
std::vector<Site> g_sites; // the hooks of the game found
u64 g_main_caller = 0;     // the return address of the main camera's call (ProjectionHandler)

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
    float depth_term; // Uncharted: -(near / 2) * far / (far - near / 2), the frame's tagged near (TaggedRendered)
    float shader_near = 0.f; // the tagged near as the shaders get it (TaggedRendered matches this)
    u32 tag = 0;      // development (BB_VR_TAGLOG): the tag, and the near and far as written
    float near_plane = 0.f, far_plane = 0.f;
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
std::atomic<float> g_game_xscale{0.f}; // the main view's x scale as the game set it (GameXScale)
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
// Frame tags: each frame's near plane is the game's plus k * NearStep (k = 1..16, cycling), so the frame's projection
// says which head pose it was rendered with (the GPU renders a frame or two behind this hook): the stereo capture reads
// it back from the motion blur's velocity pass (TaggedRendered). 16 steps are 3.2 mm at most. The step is ten times
// the far plane's effect on the shaders' depth term (U1: 1.15e-5 at a far of 5000, 2.3e-5 at 2500, against 1e-5
// steps before): passes drawn with another far matched no tag, or the wrong one (1.7% of frames).
constexpr float NearStep = 2e-4f;
constexpr u32 TagCount = 16;
u32 g_tag = 0;
float g_near_written = 0.f, g_near_original = 0.f;

// Head aim (BB_VR_HEAD_AIM=1): while the game aims (the zoomed view: the same test as the reticle layer), the head
// moves the aim. The view is the head's pose on a base orientation R that stands still while aiming (the camera's
// when the aim began), so the head looks where it looks in the world, and the aiming camera's direction is set to
// the head's at the start of its update each frame (AimHandler): the game turns that by the right stick and aims
// along it, so the camera, the reticle and Nathan's gun follow the head in the same frame. What the update added to
// the direction we wrote, sideways, is the stick's turn: R turns with it, so the stick still turns left and right
// (up and down is the head's; the game's own limit on looking up and down is not taken into R). The camera's
// direction in R's frame (AimOffset) is where the reticle goes. When the aim ends the view is the camera's again.
struct Mat3 {
    float m[9];
};
Mat3 Mul3(const Mat3& a, const Mat3& b) {
    Mat3 r{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            r.m[i * 3 + j] = a.m[i * 3] * b.m[j] + a.m[i * 3 + 1] * b.m[3 + j] + a.m[i * 3 + 2] * b.m[6 + j];
        }
    }
    return r;
}
Mat3 Transpose3(const Mat3& a) {
    Mat3 r{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            r.m[i * 3 + j] = a.m[j * 3 + i];
        }
    }
    return r;
}
float Dot3(const float* a, const float* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
constexpr Mat3 Identity3{{1, 0, 0, 0, 1, 0, 0, 0, 1}};
Mat3 g_aim_still = Identity3;  // R
float g_stick_total = 0.f; // development: the stick yaw given to the held view (BB_VR_HEAD_AIM_LOG)
std::atomic<float> g_stick_yaw{0.f}; // degrees the stick turned the aim (AimHandler) not yet in R
float g_aim_written[3] = {};          // the direction AimHandler wrote last (the update's turn is measured from it)
bool g_aim_written_valid = false;

// Rotation of the still base by the stick's turn: yaw to the right about the world's vertical (the aiming camera turns
// about it, whatever its pitch), pitch up in its own frame (x left, y up, z forward), then the rows made orthonormal
// again.
Mat3 TurnBase(const Mat3& base, float yaw_degrees, float pitch_degrees) {
    const float y = yaw_degrees * 0.0174533f, p = pitch_degrees * 0.0174533f;
    const float cp = std::cos(p), sp = std::sin(p);
    const Mat3 pitch{{1, 0, 0, 0, cp, -sp, 0, sp, cp}};
    Mat3 r = Mul3(pitch, base);
    // About the world's up (+Y), in the direction that turns the forward row toward the right (minus the left row).
    const float forward_x = r.m[6], forward_z = r.m[8];
    const float toward_right = -r.m[0] * forward_z + r.m[2] * forward_x; // (up x forward) . right, up = +Y
    const float angle = toward_right >= 0.f ? y : -y;
    const float ca = std::cos(angle), sa = std::sin(angle);
    for (int row = 0; row < 3; ++row) {
        float* v = r.m + row * 3;
        // v' = v cos + (up x v) sin + up (up . v)(1 - cos): up x v = (v.z, 0, -v.x)
        const float x = v[0] * ca + v[2] * sa, z = v[2] * ca - v[0] * sa;
        v[0] = x;
        v[2] = z;
    }
    auto normalize = [](float* v) {
        const float n = 1.f / std::sqrt(Dot3(v, v));
        v[0] *= n, v[1] *= n, v[2] *= n;
    };
    normalize(r.m + 6);
    for (int axis = 1; axis >= 0; --axis) {
        float* v = r.m + axis * 3;
        for (int other = axis + 1; other < 3; ++other) {
            const float d = Dot3(v, r.m + other * 3);
            for (int k = 0; k < 3; ++k) {
                v[k] -= d * r.m[other * 3 + k];
            }
        }
        normalize(v);
    }
    return r;
}
bool g_aim_engaged = false;
std::atomic<float> g_aim_error_right{0.f}, g_aim_error_up{0.f}, g_aim_offset_right{0.f}, g_aim_offset_up{0.f};
std::atomic<long long> g_aim_stamp{0}; // steady clock ms of the last frame engaged, 0 when not

long long NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// The base orientation for this frame, written into the rows of `in` (the camera-to-world input).
void HeadAim(const float* h, float* in, float xscale) {
    static const bool enabled = [] {
        const char* v = std::getenv("BB_VR_HEAD_AIM");
        return v && v[0] == '1';
    }();
    if (!enabled) {
        return;
    }
    static const float threshold = [] {
        const char* v = std::getenv("BB_VR_RETICLE_XSCALE");
        return v ? float(std::atof(v)) : 1.8f;
    }();
    const bool aiming = xscale > threshold;
    Mat3 camera{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            camera.m[r * 3 + c] = in[r * 4 + c];
        }
    }
    if (aiming && !g_aim_engaged) {
        g_aim_still = camera;
        g_aim_engaged = true;
        g_stick_yaw = 0.f;
        g_aim_written_valid = false;
    } else if (!aiming && g_aim_engaged) {
        g_aim_engaged = false;
        g_aim_stamp = 0;
    }
    if (g_aim_engaged) {
        const float turn_yaw = g_stick_yaw.exchange(0.f);
        g_aim_still = TurnBase(g_aim_still, turn_yaw, 0.f);
        g_stick_total += turn_yaw;
    }
    const Mat3 base = g_aim_engaged ? g_aim_still : camera;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            in[r * 4 + c] = base.m[r * 3 + c];
        }
    }
    if (!g_aim_engaged) {
        return;
    }
    Mat3 head{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            head.m[r * 3 + c] = h[r * 4 + c];
        }
    }
    constexpr float Degrees = 57.29578f;
    const Mat3 gaze = Mul3(head, base); // the view's axes in the world; z is forward
    // The head's direction in the camera's frame (x: left, y: up, z: forward).
    const float* g = gaze.m + 6;
    const float gx = Dot3(g, camera.m), gy = Dot3(g, camera.m + 3), gz = Dot3(g, camera.m + 6);
    g_aim_error_right = std::atan2(-gx, gz) * Degrees;
    g_aim_error_up = std::atan2(gy, std::hypot(gx, gz)) * Degrees;
    // The camera's direction in the still base's frame: where the reticle is.
    const float* a = camera.m + 6;
    const float ax = Dot3(a, base.m), ay = Dot3(a, base.m + 3), az = Dot3(a, base.m + 6);
    g_aim_offset_right = std::atan2(-ax, az) * Degrees;
    g_aim_offset_up = std::atan2(ay, std::hypot(ax, az)) * Degrees;
    g_aim_stamp = NowMs();
    static const bool log = std::getenv("BB_VR_HEAD_AIM_LOG") != nullptr;
    static int frames = 0;
    static const int interval = std::max(1, std::atoi(std::getenv("BB_VR_HEAD_AIM_LOG") ? std::getenv("BB_VR_HEAD_AIM_LOG") : "1"));
    if (log && frames++ % interval == 0) {
        std::printf("VR head aim: stick %.2f gaze %.2f right %.2f up of the camera, reticle %.2f right %.2f up of the view\n",
                    g_stick_total, g_aim_error_right.load(), g_aim_error_up.load(), g_aim_offset_right.load(),
                    g_aim_offset_up.load());
        std::fflush(stdout);
    }
}

// The aiming camera's update begins: the head's direction goes in (see HeadAim), after the turn the last update
// gave the direction written then is taken into R.
extern "C" void AimHandler(u64* regs) {
    if (!g_aim_engaged || !g_game) {
        g_aim_written_valid = false;
        return;
    }
    auto* direction = reinterpret_cast<float*>(regs[RSI] + g_game->aim_direction);
    if (g_aim_written_valid) {
        // The sideways turn from what was written to what the update left: about the world's up, right positive.
        const float* a = g_aim_written;
        const float* b = direction;
        float ax = a[0], az = a[2];
        const float an = std::hypot(ax, az);
        if (an > 1e-4f) {
            ax /= an, az /= an;
            float rx = -g_aim_still.m[0], rz = -g_aim_still.m[2]; // R's right, level
            const float along = rx * ax + rz * az;
            rx -= along * ax, rz -= along * az;
            const float rn = std::hypot(rx, rz);
            if (rn > 1e-4f) {
                rx /= rn, rz /= rn;
                g_stick_yaw.fetch_add(std::atan2(b[0] * rx + b[2] * rz, b[0] * ax + b[2] * az) * 57.29578f);
            }
        }
    }
    // The head's direction: its pose on R (rows: the head's axes in the game's frame, z forward).
    Head head;
    {
        std::scoped_lock lock{g_head_mutex};
        if (!g_head_valid) {
            return;
        }
        head = g_head;
    }
    constexpr float axes[3] = {-1.f, 1.f, -1.f};
    const Mat3 still = TurnBase(g_aim_still, g_stick_yaw.load(), 0.f);
    float forward[3];
    for (int c = 0; c < 3; ++c) {
        forward[c] = 0.f;
        for (int k = 0; k < 3; ++k) {
            forward[c] += head.pose[2 * 4 + k] * axes[2] * axes[k] * still.m[k * 3 + c];
        }
    }
    std::memcpy(direction, forward, sizeof(forward));
    std::memcpy(g_aim_written, forward, sizeof(forward));
    g_aim_written_valid = true;
}

extern "C" void ProjectionHandler(u64* regs) {
    auto* view = reinterpret_cast<u8*>(regs[RDI]);
    const Game& game = *g_game;
    auto* input = reinterpret_cast<float*>(view + game.input);   // rows x, y, z, position (w unused)
    auto* params = reinterpret_cast<float*>(view + game.params); // aspect, x scale, near, far
    if (regs[RegCount] != g_main_caller || !(params[1] > 0.1f) || !(params[2] > 0.05f && params[2] < 1.f)) {
        return;
    }
    if (g_near_written != 0.f && params[2] == g_near_written) {
        params[2] = g_near_original; // the game did not set it again this frame
    }
    g_near = params[2];
    g_game_xscale = params[1];
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
    HeadAim(h, in, params[1]);
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
    g_near_original = params[2];
    g_tag = g_tag % TagCount + 1;
    params[2] = g_near_original + float(g_tag) * NearStep;
    g_near_written = params[2];
    const float n = params[2], f = params[3];
    std::scoped_lock lock{g_head_mutex};
    g_rendered.tan[0] = -tx, g_rendered.tan[1] = tx, g_rendered.tan[2] = ty, g_rendered.tan[3] = -ty;
    Rendered& entry = g_history[g_history_count++ % HistorySize];
    std::memcpy(entry.position, out + 12, sizeof(entry.position));
    entry.head = g_rendered;
    // The near plane as the shaders get it (U1: half the view's: 0.48 in the view, 0.24 in the velocity pass).
    const float sn = n * game.shader_near;
    entry.depth_term = f > sn ? -sn * f / (f - sn) : 0.f;
    entry.shader_near = f > sn ? sn : 0.f;
    entry.tag = g_tag;
    entry.near_plane = n;
    entry.far_plane = f;
    static int told = 0; // development: the first tags written
    if (told < 12 && std::getenv("BB_VR_TAGLOG")) {
        ++told;
        std::printf("VR tag: frame tag %u near %.7f far %.3f term %.7f%c", g_tag, n, f, entry.depth_term, 10);
    }
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
        dq(site.address + site.bytes.size());
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
    for (const Site& site : g_sites) {
        auto* code = reinterpret_cast<u8*>(site.address);
        MEMORY_BASIC_INFORMATION mbi{};
        if (!VirtualQuery(code, &mbi, sizeof(mbi)) || mbi.State != MEM_COMMIT ||
            std::memcmp(code, site.bytes.data(), site.bytes.size()) != 0) {
            std::printf("VR camera: the view setter at %#llx is not as expected: camera hooks off\n",
                        (unsigned long long)site.address);
            std::fflush(stdout);
            return;
        }
    }
    const bool cull_head = !(std::getenv("BB_VR_CULL_HEAD") && std::getenv("BB_VR_CULL_HEAD")[0] == '0');
    for (const Site& site : g_sites) {
        if (site.handler == &CameraSetHandler && !cull_head) {
            continue;
        }
        auto* stub = new Stub(site);
        stub->ready();
        g_stubs.push_back(stub);
        auto* code = reinterpret_cast<u8*>(site.address);
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
    // The HUD's canvas: the game's render size (square at a VR eye size: the HUD lays itself out for it).
    u32 width = 0, height = 0;
    return Fixes::RenderSize(width, height) && height ? float(width) / float(height) : 16.f / 9.f;
}

// An unsupported game version: where the opening instructions of each known game's projection update are in the
// executable allocations, if anywhere, for the launcher's report (the start of a new Games entry's pattern).
void ReportUnknownImage() {
#ifdef _WIN32
    MEMORY_BASIC_INFORMATION mbi{};
    size_t hits = 0;
    for (u64 at = 0x400000; at < 0x80000000ull && VirtualQuery(reinterpret_cast<void*>(at), &mbi, sizeof(mbi));
         at = u64(mbi.BaseAddress) + mbi.RegionSize) {
        const u64 base = u64(mbi.BaseAddress);
        if (mbi.State != MEM_COMMIT || u64(mbi.AllocationBase) != base || mbi.RegionSize < 0x100000 ||
            !(mbi.Protect & (PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE))) {
            continue;
        }
        std::printf("VR camera: executable image at %#llx, %#llx bytes%c", (unsigned long long)base,
                    (unsigned long long)mbi.RegionSize, 10);
        for (const Game& game : Games) {
            Sig opening{game.func};
            opening.bytes.resize(game.displaced);
            for (u64 i : Find(reinterpret_cast<const u8*>(base), mbi.RegionSize, opening, 24 - std::min<size_t>(hits, 23))) {
                std::printf("VR camera: opening of %s's projection update at image +%#llx%c", game.name,
                            (unsigned long long)i, 10);
                ++hits;
            }
        }
    }
#endif
}

// The game image is mapped before the first frame: the executable allocation holding a known game's projection
// update (its pattern matching exactly once), then the camera update's call to it (once, calling that function).
bool Locate(const u8* code, u64 size, const Game& game) {
    const Sig func{game.func};
    const Sig call{game.caller};
    const auto found = Find(code, size, func, 2);
    u64 function = found.size() == 1 ? reinterpret_cast<u64>(code) + found[0] : 0;
    if (!function) {
        // A version whose function differs past its opening: the camera update's call (unique) gives its address.
        const auto calls = Find(code, size, call, 2);
        if (calls.size() != 1) {
            return false;
        }
        const u64 at = reinterpret_cast<u64>(code) + calls[0] + call.bytes.size() - 5;
        int32_t rel;
        std::memcpy(&rel, reinterpret_cast<const void*>(at + 1), 4);
        Sig opening{game.func};
        opening.bytes.resize(game.displaced);
        function = at + 5 + rel;
        if (function < reinterpret_cast<u64>(code) || function + game.displaced > reinterpret_cast<u64>(code) + size ||
            !opening.At(reinterpret_cast<const u8*>(function))) {
            return false;
        }
    }
    u64 caller = 0;
    for (u64 i : Find(code, size, call, 64)) {
        const u64 at = reinterpret_cast<u64>(code) + i + call.bytes.size() - 5; // the e8
        int32_t rel;
        std::memcpy(&rel, reinterpret_cast<const void*>(at + 1), 4);
        if (at + 5 + rel == function) {
            if (caller) {
                caller = 0;
                break; // more than one: not the pattern of this version
            }
            caller = at + 5;
        }
    }
    if (!caller) {
        std::printf("VR camera: %s's projection update is at %#llx, but its main camera call was not found%c",
                    game.name, (unsigned long long)function, 10);
        std::fflush(stdout);
        return false;
    }
    Site site;
    site.address = function;
    site.bytes.assign(reinterpret_cast<const u8*>(function), reinterpret_cast<const u8*>(function) + game.displaced);
    site.handler = &ProjectionHandler;
    g_sites = {site};
    if (game.aim) {
        const auto aim = Find(code, size, Sig{game.aim}, 2);
        if (aim.size() == 1) {
            Site update;
            update.address = reinterpret_cast<u64>(code) + aim[0];
            update.bytes.assign(reinterpret_cast<const u8*>(update.address),
                                reinterpret_cast<const u8*>(update.address) + game.aim_displaced);
            update.handler = &AimHandler;
            g_sites.push_back(update);
        } else {
            std::printf("VR camera: the aiming camera's update was not found: no head aim%c", 10);
        }
    }
    g_main_caller = caller;
    g_game = &game;
    std::printf("VR camera: %s, projection update at image +%#llx, main camera call at +%#llx%c", game.name,
                (unsigned long long)(function - reinterpret_cast<u64>(code)),
                (unsigned long long)(caller - reinterpret_cast<u64>(code)), 10);
    std::fflush(stdout);
    return true;
}

void Install() {
#ifdef _WIN32
    // Looked for every 60th frame (the game unpaced at boot flips hundreds a second: no deadline).
    if (!g_image_base) {
        static int calls = 0;
        static const auto started = std::chrono::steady_clock::now();
        static bool reported = false;
        if (calls++ % 60 != 0) {
            return;
        }
        MEMORY_BASIC_INFORMATION mbi{};
        for (u64 at = 0x400000; at < 0x80000000ull && VirtualQuery(reinterpret_cast<void*>(at), &mbi, sizeof(mbi)) &&
                                !g_image_base;
             at = u64(mbi.BaseAddress) + mbi.RegionSize) {
            const u64 base = u64(mbi.BaseAddress);
            if (mbi.State != MEM_COMMIT || u64(mbi.AllocationBase) != base || mbi.RegionSize < 0x100000 ||
                !(mbi.Protect & (PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE))) {
                continue;
            }
            for (const Game& game : Games) {
                if (Locate(reinterpret_cast<const u8*>(base), mbi.RegionSize, game)) {
                    g_image_base = base;
                    break;
                }
            }
        }
        if (!g_image_base) {
            if (!reported && std::chrono::steady_clock::now() - started > std::chrono::seconds(60)) {
                reported = true;
                std::printf("VR camera: no known game image yet (the game on a flat screen until then)%c", 10);
                ReportUnknownImage();
                std::fflush(stdout);
            }
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

bool AimOffset(float& right, float& up) {
    const long long stamp = g_aim_stamp.load();
    if (!stamp || NowMs() - stamp > 250) {
        return false;
    }
    right = g_aim_offset_right.load();
    up = g_aim_offset_up.load();
    return true;
}

float GameXScale() {
    return g_game_xscale.load();
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

// Matched on the near plane itself (-c9 / c8 of the velocity pass): the depth term alone also moves with the far plane,
// and frames drawn with another far (2500 of 5000) shifted it by a whole tag step (1.7% of frames found no tag, and a
// shift of exactly one step would match the frame before). Some velocity passes carry c8 = 1 (the near is then the
// term's): those match on the depth term as before.
bool TaggedRendered(float shader_near, float depth_term, float (&layer_pose)[7], float (&tangents)[4]) {
    std::scoped_lock lock{g_head_mutex};
    const u32 count = std::min(g_history_count, HistorySize);
    const Rendered* best = nullptr;
    float best_error = 0.f;
    for (int pass = shader_near != 0.f ? 0 : 1; pass < 2; ++pass) { // the near plane, else (c8 = 1) the depth term
        best = nullptr;
        for (u32 n = 0; n < count; ++n) { // oldest to newest: the newest of equal matches wins
            const Rendered& e = g_history[(g_history_count - count + n) % HistorySize];
            const float have = pass ? e.depth_term : e.shader_near, want = pass ? depth_term : shader_near;
            const float error = std::abs(have - want);
            if (have != 0.f && (!best || error <= best_error)) {
                best = &e;
                best_error = error;
            }
        }
        break;
    }
    // The tags are NearStep * shader_near apart in the shaders' term.
    if (!best || best_error > NearStep * (g_game ? g_game->shader_near : 1.f) * 0.4f) {
        static const bool log = std::getenv("BB_VR_TAGLOG") != nullptr; // development: why a frame found no tag
        static int told = 0;
        if (log && told < 60) {
            ++told;
            std::string text;
            for (u32 n = 0; n < std::min(count, 6u); ++n) { // the newest six
                const Rendered& e = g_history[(g_history_count - 1 - n) % HistorySize];
                char row[96];
                std::snprintf(row, sizeof(row), " [age %u tag %u near %.6f far %.1f shader near %.7f]", n, e.tag,
                              e.near_plane, e.far_plane, e.shader_near);
                text += row;
            }
            std::printf("VR tag miss: read %.7f, best error %.2e (tag %u);%s%c", depth_term, best_error,
                        best ? best->tag : 0, text.c_str(), 10);
            std::fflush(stdout);
        }
        return false;
    }
    std::memcpy(layer_pose, best->head.layer, sizeof(layer_pose));
    std::memcpy(tangents, best->head.tan, sizeof(tangents));
    return true;
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


