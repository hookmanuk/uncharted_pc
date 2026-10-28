// SPDX-License-Identifier: GPL-2.0-or-later
// VR (dcport-vr): OpenXR session, frame loop and the floating screen. See vr_openxr.h.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <string>
#include <vector>

#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vr/vr_camera.h"
#include "video_core/renderer_vulkan/vr/vr_multiview.h"
#include "video_core/renderer_vulkan/vr/vr_dev.h"
#include "video_core/renderer_vulkan/vr/vr_fixes.h"
#include "video_core/renderer_vulkan/vr/vr_openxr.h"
#include "video_core/renderer_vulkan/vr/vr_stereo.h"

#define XR_USE_GRAPHICS_API_VULKAN
#include "../../../../third_party/openxr/include/openxr/openxr.h"
#include "../../../../third_party/openxr/include/openxr/openxr_platform.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace Vulkan::Vr {
namespace {

void Log(const char* format, ...) {
    va_list args;
    va_start(args, format);
    std::fputs("VR: ", stdout);
    std::vprintf(format, args);
    std::fputc('\n', stdout);
    std::fflush(stdout);
    va_end(args);
}

struct Functions {
    PFN_xrGetInstanceProcAddr GetInstanceProcAddr{};
    PFN_xrGetDisplayRefreshRateFB GetDisplayRefreshRateFB{}; // XR_FB_display_refresh_rate, when the runtime has it
#define XR_FN(name) PFN_xr##name name{};
#define XR_FUNCTIONS                                                                                                   \
    XR_FN(CreateInstance)                                                                                              \
    XR_FN(EnumerateInstanceExtensionProperties)                                                                        \
    XR_FN(DestroyInstance)                                                                                             \
    XR_FN(GetSystem)                                                                                                   \
    XR_FN(GetVulkanInstanceExtensionsKHR)                                                                              \
    XR_FN(GetVulkanDeviceExtensionsKHR)                                                                                \
    XR_FN(GetVulkanGraphicsDeviceKHR)                                                                                  \
    XR_FN(GetVulkanGraphicsRequirementsKHR)                                                                            \
    XR_FN(CreateSession)                                                                                               \
    XR_FN(BeginSession)                                                                                                \
    XR_FN(EndSession)                                                                                                  \
    XR_FN(PollEvent)                                                                                                   \
    XR_FN(CreateReferenceSpace)                                                                                        \
    XR_FN(EnumerateSwapchainFormats)                                                                                   \
    XR_FN(CreateSwapchain)                                                                                             \
    XR_FN(DestroySwapchain)                                                                                            \
    XR_FN(EnumerateSwapchainImages)                                                                                    \
    XR_FN(AcquireSwapchainImage)                                                                                       \
    XR_FN(WaitSwapchainImage)                                                                                          \
    XR_FN(ReleaseSwapchainImage)                                                                                       \
    XR_FN(WaitFrame)                                                                                                   \
    XR_FN(BeginFrame)                                                                                                  \
    XR_FN(EndFrame)                                                                                                    \
    XR_FN(LocateViews)                                                                                                 \
    XR_FN(LocateSpace)
    XR_FUNCTIONS
#undef XR_FN
};

struct State {
    std::once_flag init_once;
    Functions xr;
    XrInstance instance{XR_NULL_HANDLE};
    XrSystemId system{XR_NULL_SYSTEM_ID};
    std::string instance_extension_text, device_extension_text;
    std::vector<const char*> instance_extensions, device_extensions;
    bool device_extensions_queried = false;

    VkDevice device{};
    VkPhysicalDevice physical_device{};
    XrSession session{XR_NULL_HANDLE};
    XrSpace local_space{XR_NULL_HANDLE};
    XrSpace view_space{XR_NULL_HANDLE};
    XrSessionState session_state{XR_SESSION_STATE_UNKNOWN};
    bool running = false;
    std::atomic<bool> recenter_requested{false};
    // Recentring: the head's position and yaw at the first tracked frame (or a RECENTER request) are the seat's origin.
    bool have_reference = false;
    float reference_yaw = 0.f;
    XrVector3f reference_position{};

    // The floating screen: an OpenXR swapchain (the headset's thread) and three images in its UNORM twin (the presenter
    // blits the frame into one, scaled and in the swapchain's channel order; it is copied bit for bit into the sRGB
    // swapchain image).
    XrSwapchain swapchain{XR_NULL_HANDLE};
    u32 width = 0, height = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    bool rgba_offered = false; // R8G8B8A8_SRGB swapchains (stereo frames whose bytes are RGBA: Uncharted)
    std::vector<XrSwapchainImageVulkanKHR> images;
    VkImage screen_images[3]{};
    VkDeviceMemory screen_memory[3]{};
    u32 screen_width = 0, screen_height = 0, screen_next = 0;

    // The headset's frame thread and its own queue (a second queue of the game's family): the runtime waits for the GPU
    // inside its calls (SteamVR's Vulkan path), which held the game's presentation up and took its frame rate when the
    // frame loop ran on the presenter. Without a second queue the runtime shares the game's under the submit lock.
    std::thread thread;
    std::atomic<bool> quit{false};
    u32 queue_count = 1;
    u32 queue_index = 0; // the headset's queue in the game's family (past the queues the renderer uses)
    bool own_queue = false;
    VkQueue queue{};
    VkCommandPool pool{};
    VkCommandBuffer cmds[3]{};
    VkFence fences[3]{};
    u32 next_cmd = 0;
    std::mutex frame_mutex; // held by the headset's thread for a frame (not while it waits for the runtime)

    // What the presenter publishes for the headset: a stereo frame (a slot and its pose) or a screen image, ready on the
    // GPU once `semaphore` reaches `tick` (the presentation's submission).
    enum class Content { None, Stereo, Screen };
    struct Published {
        Content kind = Content::None;
        Stereo::Frame stereo{};
        float layer_pose[7]{}, tangents[4]{};
        u32 screen = 0, screen_width = 0, screen_height = 0;
        bool movie = false; // a screen frame with a movie (fitted into the render's shape: Fixes::MovieCanvas)
        VkSemaphore semaphore{};
        u64 tick = 0;
        u64 serial = 0;
    };
    Published pending;          // the presenter's, this presentation
    std::mutex publish_mutex;
    Published published;        // under publish_mutex
    // The latest publications, newest first (under publish_mutex): the headset shows the newest whose GPU work is
    // done. Stereo slots are four, so three publications are still intact; screen images three, so two.
    std::array<Published, 3> recent{};
    std::condition_variable publish_cv; // a new publication (SubmitFrame), for the headset's wait (WaitForGameFrame)
    // The headset's loop, per report: time in xrWaitFrame, xrBeginFrame (with the frame lock), Show, xrEndFrame.
    std::array<double, 5> stage_sum{}, stage_max{}; // [4]: the wait for the game's frame (VR fork)
    // Show's parts (stereo): acquire the three swapchain images, wait for a command buffer, record + submit, release.
    std::array<double, 5> part_sum{}, part_max{};
    // The adaptive frame cap (FramePeriod): the headset's period (ns, from xrWaitFrame) and the game's flips.
    std::atomic<u64> display_period_ns{0};
    // Uncharted fork: the headset's refresh period (XR_FB_display_refresh_rate, else the shortest display period seen).
    // The predicted display period doubles while SteamVR holds a slow game at half rate: capping the game at that
    // (45 Hz, the game making 54) tied the output to the runtime's throttle, not the game's render rate.
    std::atomic<u64> refresh_period_ns{0};
    std::atomic<bool> capped{false};
    std::atomic<bool> flip_token{false}; // set by each headset frame, taken by a capped flip
    std::chrono::steady_clock::time_point flips_since{};
    u32 flips = 0;
    u64 shown_serial = 0;       // the headset's thread: the last publication shown
    Content content = Content::None; // the headset's thread: what the swapchains hold
    bool shown_movie = false;          // the headset's thread: the shown screen frame has a movie

    // Stereo: an eye swapchain each and the HUD's (sRGB, the scene's size).
    struct Chain {
        XrSwapchain swapchain{XR_NULL_HANDLE};
        std::vector<XrSwapchainImageVulkanKHR> images;
        u32 width = 0, height = 0;
        u32 index = 0;
        VkFormat format = VK_FORMAT_UNDEFINED;
    };
    Chain eye_chains[2], hud_chain, far_chain; // far: the reticle layer (Stereo::ReticleDistance)
    int frames_since_stereo = 1000; // the presenter: frames since the last stereo frame
    // The headset's thread: when the flat screen took over (a loading screen); stereo again after a long one is a new
    // race (or menu) beginning, and the seat is recentred there.
    std::chrono::steady_clock::time_point screen_since{};
    // Frame statistics (submitted headset frames).
    std::vector<double> intervals;
    double last_submit = 0.0;
    u32 repeats = 0, screens = 0;
    std::atomic<u32> count_publications{0}; // stereo frames published to the headset since the last report
    std::atomic<u32> count_flips{0};        // the game's flips (RecordFrame: one per presented game frame)
    // Development (BB_VR_POSELOG=<frames>): per headset frame, the head's yaw at its display time, the submitted
    // layer's, whether the frame was new and whether its pose came from its frame tag (judder in a real headset).
    float log_head_yaw = 0.f;
    bool log_new_frame = false, log_tagged = false;
    u64 log_serial = 0;
    float frame_layer_pose[7]{}, frame_tangents[4]{}; // the stereo frame's rendered pose (its layer)
    float ipd = 0.064f;

    XrFrameState frame_state{XR_TYPE_FRAME_STATE}; // the headset's thread
};

State g;
std::mutex g_flip_token_mutex; // the present thread waits on the headset's frame token (WaitFlipAllowed)
std::condition_variable g_flip_token_cv;

bool Check(XrResult result, const char* what) {
    if (XR_SUCCEEDED(result)) {
        return true;
    }
    Log("%s failed (%d)", what, int(result));
    return false;
}

// The runtime's calls may submit on the queue it was given: alone on its own queue, under the game's submit lock when
// it shares the game's.
struct QueueLock {
    std::unique_lock<std::mutex> lock;
    QueueLock() {
        if (!g.own_queue) {
            lock = std::unique_lock{Scheduler::submit_mutex};
        }
    }
};

bool LoadLoader() {
#ifdef _WIN32
    wchar_t exe[MAX_PATH]{};
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::wstring path(exe);
    path = path.substr(0, path.find_last_of(L"\\/") + 1) + L"openxr_loader.dll";
    HMODULE module = LoadLibraryW(path.c_str());
    if (!module) {
        module = LoadLibraryW(L"openxr_loader.dll");
    }
    if (!module) {
        return false;
    }
    g.xr.GetInstanceProcAddr =
        reinterpret_cast<PFN_xrGetInstanceProcAddr>(GetProcAddress(module, "xrGetInstanceProcAddr"));
#else
    void* module = dlopen("libopenxr_loader.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!module) {
        return false;
    }
    g.xr.GetInstanceProcAddr = reinterpret_cast<PFN_xrGetInstanceProcAddr>(dlsym(module, "xrGetInstanceProcAddr"));
#endif
    return g.xr.GetInstanceProcAddr != nullptr;
}

void LoadFunctions(XrInstance instance) {
#define XR_FN(name)                                                                                                    \
    g.xr.GetInstanceProcAddr(instance, "xr" #name, reinterpret_cast<PFN_xrVoidFunction*>(&g.xr.name));
    XR_FUNCTIONS
#undef XR_FN
}

std::vector<const char*> Split(std::string& text) {
    std::vector<const char*> names;
    for (char& c : text) {
        if (c == ' ') {
            c = '\0';
        }
    }
    for (size_t i = 0; i < text.size();) {
        const size_t len = std::strlen(text.c_str() + i);
        if (len) {
            names.push_back(text.c_str() + i);
        }
        i += len + 1;
    }
    return names;
}

void Init() {
    // Opt-in (the launcher's VR row sets BB_VR=1): flat play never starts the OpenXR runtime.
    if (const char* v = std::getenv("BB_VR"); !v || v[0] != '1') {
        return;
    }
    if (!LoadLoader()) {
        Log("no openxr_loader.dll: VR off");
        return;
    }
    LoadFunctions(XR_NULL_HANDLE); // xrCreateInstance and friends work without an instance
    // Uncharted fork: the headset's refresh rate (XR_FB_display_refresh_rate, SteamVR has it) for the frame cap, as the
    // predicted display period doubles while the runtime holds the game at half rate.
    bool refresh_rate = false;
    if (g.xr.EnumerateInstanceExtensionProperties) {
        u32 count = 0;
        g.xr.EnumerateInstanceExtensionProperties(nullptr, 0, &count, nullptr);
        std::vector<XrExtensionProperties> offered(count, {XR_TYPE_EXTENSION_PROPERTIES});
        g.xr.EnumerateInstanceExtensionProperties(nullptr, count, &count, offered.data());
        for (const auto& extension : offered) {
            refresh_rate |= std::strcmp(extension.extensionName, XR_FB_DISPLAY_REFRESH_RATE_EXTENSION_NAME) == 0;
        }
    }
    const char* extensions[] = {XR_KHR_VULKAN_ENABLE_EXTENSION_NAME, XR_FB_DISPLAY_REFRESH_RATE_EXTENSION_NAME};
    XrInstanceCreateInfo info{XR_TYPE_INSTANCE_CREATE_INFO};
    std::strcpy(info.applicationInfo.applicationName, "Uncharted VR");
    std::strcpy(info.applicationInfo.engineName, "Uncharted PC port");
    info.applicationInfo.apiVersion = XR_MAKE_VERSION(1, 0, 0);
    info.enabledExtensionCount = refresh_rate ? 2 : 1;
    info.enabledExtensionNames = extensions;
    if (!g.xr.CreateInstance || !Check(g.xr.CreateInstance(&info, &g.instance), "xrCreateInstance")) {
        g.instance = XR_NULL_HANDLE;
        return;
    }
    LoadFunctions(g.instance);
    if (refresh_rate) {
        g.xr.GetInstanceProcAddr(g.instance, "xrGetDisplayRefreshRateFB",
                                 reinterpret_cast<PFN_xrVoidFunction*>(&g.xr.GetDisplayRefreshRateFB));
    }
    XrSystemGetInfo system_info{XR_TYPE_SYSTEM_GET_INFO};
    system_info.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
    if (!Check(g.xr.GetSystem(g.instance, &system_info, &g.system), "xrGetSystem (no headset?)")) {
        g.xr.DestroyInstance(g.instance);
        g.instance = XR_NULL_HANDLE;
        return;
    }
    u32 size = 0;
    g.xr.GetVulkanInstanceExtensionsKHR(g.instance, g.system, 0, &size, nullptr);
    g.instance_extension_text.resize(size);
    g.xr.GetVulkanInstanceExtensionsKHR(g.instance, g.system, size, &size, g.instance_extension_text.data());
    g.instance_extension_text.resize(size ? size - 1 : 0);
    g.instance_extensions = Split(g.instance_extension_text);
    Log("OpenXR instance ready; Vulkan instance extensions: %s", g.instance_extension_text.c_str());
}

VkFormat UnormTwin(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R8G8B8A8_SRGB:
        return VK_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_B8G8R8A8_SRGB:
        return VK_FORMAT_B8G8R8A8_UNORM;
    default:
        return format;
    }
}

// Before freeing images or swapchains the GPU may still be using (a resize): wait for the device. The caller holds
// frame_mutex or is the headset's thread (its queue idle meanwhile); the game's queue is held by the submit lock.
void WaitIdle() {
    std::scoped_lock lock{Scheduler::submit_mutex};
    VULKAN_HPP_DEFAULT_DISPATCHER.vkDeviceWaitIdle(g.device);
}

void DestroySwapchain() {
    if (g.swapchain) {
        WaitIdle();
    }
    if (g.swapchain) {
        QueueLock lock;
        g.xr.DestroySwapchain(g.swapchain);
        g.swapchain = XR_NULL_HANDLE;
    }
    g.images.clear();
}

// The swapchain formats: 8-bit sRGB (the frames are gamma-encoded already and copied bit for bit).
bool PickFormat() {
    u32 count = 0;
    g.xr.EnumerateSwapchainFormats(g.session, 0, &count, nullptr);
    std::vector<int64_t> formats(count);
    g.xr.EnumerateSwapchainFormats(g.session, count, &count, formats.data());
    g.format = VK_FORMAT_UNDEFINED;
    for (const VkFormat wanted : {VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_R8G8B8A8_SRGB}) {
        for (const int64_t f : formats) {
            if (g.format == VK_FORMAT_UNDEFINED && VkFormat(f) == wanted) {
                g.format = wanted;
            }
            g.rgba_offered |= VkFormat(f) == VK_FORMAT_R8G8B8A8_SRGB;
        }
    }
    if (g.format == VK_FORMAT_UNDEFINED) {
        Log("the runtime offers no 8-bit sRGB swapchain format");
        return false;
    }
    return true;
}

// The headset's thread: the screen's swapchain at the screen images' size.
bool CreateSwapchain(u32 width, u32 height) {
    DestroySwapchain();
    XrSwapchainCreateInfo info{XR_TYPE_SWAPCHAIN_CREATE_INFO};
    info.usageFlags = XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
    info.format = g.format;
    info.sampleCount = 1;
    info.width = width;
    info.height = height;
    info.faceCount = 1;
    info.arraySize = 1;
    info.mipCount = 1;
    {
        QueueLock lock;
        if (!Check(g.xr.CreateSwapchain(g.session, &info, &g.swapchain), "xrCreateSwapchain")) {
            return false;
        }
    }
    u32 count = 0;
    g.xr.EnumerateSwapchainImages(g.swapchain, 0, &count, nullptr);
    g.images.assign(count, XrSwapchainImageVulkanKHR{XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR});
    g.xr.EnumerateSwapchainImages(g.swapchain, count, &count,
                                  reinterpret_cast<XrSwapchainImageBaseHeader*>(g.images.data()));
    g.width = width;
    g.height = height;
    Log("screen swapchain %ux%u, VkFormat %d, %u images", width, height, int(g.format), count);
    return true;
}

// The presenter: the screen images at the frame's size (UNORM twin of the swapchain format).
bool MakeScreens(u32 width, u32 height) {
    if (g.screen_images[0] && g.screen_width == width && g.screen_height == height) {
        return true;
    }
    // The headset's thread may be copying one: it waits until the new images exist (with the old ones destroyed and
    // the new not made yet it copied from a null image, a crash in the driver).
    std::unique_lock frame_lock{g.frame_mutex, std::defer_lock};
    if (g.screen_images[0]) {
        frame_lock.lock();
        WaitIdle();
        for (u32 i = 0; i < 3; ++i) {
            VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyImage(g.device, g.screen_images[i], nullptr);
            VULKAN_HPP_DEFAULT_DISPATCHER.vkFreeMemory(g.device, g.screen_memory[i], nullptr);
            g.screen_images[i] = {};
            g.screen_memory[i] = {};
        }
    }
    VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = UnormTwin(g.format);
    image_info.extent = {width, height, 1};
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    VkPhysicalDeviceMemoryProperties memory{};
    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceMemoryProperties(g.physical_device, &memory);
    for (u32 i = 0; i < 3; ++i) {
        if (VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateImage(g.device, &image_info, nullptr, &g.screen_images[i]) != VK_SUCCESS) {
            g.screen_images[i] = {};
            return false;
        }
        VkMemoryRequirements requirements{};
        VULKAN_HPP_DEFAULT_DISPATCHER.vkGetImageMemoryRequirements(g.device, g.screen_images[i], &requirements);
        u32 type = 0;
        for (; type < memory.memoryTypeCount; ++type) {
            if ((requirements.memoryTypeBits & (1u << type)) &&
                (memory.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                break;
            }
        }
        VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc.allocationSize = requirements.size;
        alloc.memoryTypeIndex = type;
        if (VULKAN_HPP_DEFAULT_DISPATCHER.vkAllocateMemory(g.device, &alloc, nullptr, &g.screen_memory[i]) != VK_SUCCESS) {
            return false;
        }
        VULKAN_HPP_DEFAULT_DISPATCHER.vkBindImageMemory(g.device, g.screen_images[i], g.screen_memory[i], 0);
    }
    g.screen_width = width;
    g.screen_height = height;
    return true;
}

bool MakeChain(State::Chain& chain, u32 width, u32 height, bool rgba = false) {
    const VkFormat format = rgba && g.rgba_offered ? VK_FORMAT_R8G8B8A8_SRGB
                            : g.format != VK_FORMAT_UNDEFINED ? g.format
                                                              : VK_FORMAT_R8G8B8A8_SRGB;
    if (chain.swapchain && chain.width == width && chain.height == height && chain.format == format) {
        return true;
    }
    if (chain.swapchain) {
        WaitIdle();
        QueueLock lock;
        g.xr.DestroySwapchain(chain.swapchain);
        chain.swapchain = XR_NULL_HANDLE;
    }
    XrSwapchainCreateInfo info{XR_TYPE_SWAPCHAIN_CREATE_INFO};
    info.usageFlags = XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
    info.format = format;
    chain.format = format;
    info.sampleCount = 1;
    info.width = width;
    info.height = height;
    info.faceCount = 1;
    info.arraySize = 1;
    info.mipCount = 1;
    {
        QueueLock lock;
        if (!Check(g.xr.CreateSwapchain(g.session, &info, &chain.swapchain), "xrCreateSwapchain (stereo)")) {
            chain.swapchain = XR_NULL_HANDLE;
            return false;
        }
    }
    u32 count = 0;
    g.xr.EnumerateSwapchainImages(chain.swapchain, 0, &count, nullptr);
    chain.images.assign(count, XrSwapchainImageVulkanKHR{XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR});
    g.xr.EnumerateSwapchainImages(chain.swapchain, count, &count,
                                  reinterpret_cast<XrSwapchainImageBaseHeader*>(chain.images.data()));
    chain.width = width;
    chain.height = height;
    Log("stereo swapchain %ux%u, VkFormat %d", width, height, int(info.format));
    return true;
}

bool AcquireChain(State::Chain& chain) {
    // Runtimes (SteamVR) submit on the game's queue inside the swapchain image calls: one submitter at a time.
    QueueLock lock;
    XrSwapchainImageAcquireInfo acquire{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
    if (!Check(g.xr.AcquireSwapchainImage(chain.swapchain, &acquire, &chain.index), "xrAcquireSwapchainImage")) {
        return false;
    }
    XrSwapchainImageWaitInfo wait{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
    wait.timeout = XR_INFINITE_DURATION;
    g.xr.WaitSwapchainImage(chain.swapchain, &wait);
    return true;
}

// The headset's refresh rate (XR_FB_display_refresh_rate), for the frame cap (NoteFlip). Without the extension the cap
// takes the shortest display period seen.
void ReadRefreshRate() {
    float hz = 0.f;
    if (g.xr.GetDisplayRefreshRateFB && g.session &&
        XR_SUCCEEDED(g.xr.GetDisplayRefreshRateFB(g.session, &hz)) && hz > 10.f) {
        g.refresh_period_ns.store(u64(1e9 / double(hz)), std::memory_order_relaxed);
        Log("headset refresh rate %.1f Hz", hz);
    }
}

void PollEvents() {
    XrEventDataBuffer event{XR_TYPE_EVENT_DATA_BUFFER};
    while (g.xr.PollEvent(g.instance, &event) == XR_SUCCESS) {
        if (event.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
            const auto& changed = reinterpret_cast<const XrEventDataSessionStateChanged&>(event);
            g.session_state = changed.state;
            Log("session state %d", int(changed.state));
            if (changed.state == XR_SESSION_STATE_READY) {
                XrSessionBeginInfo begin{XR_TYPE_SESSION_BEGIN_INFO};
                begin.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
                g.running = Check(g.xr.BeginSession(g.session, &begin), "xrBeginSession");
                ReadRefreshRate();
            } else if (changed.state == XR_SESSION_STATE_STOPPING) {
                g.xr.EndSession(g.session);
                g.running = false;
            }
        } else if (event.type == XR_TYPE_EVENT_DATA_DISPLAY_REFRESH_RATE_CHANGED_FB) {
            ReadRefreshRate();
        } else if (event.type == XR_TYPE_EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING) {
            // The runtime moved LOCAL (a seated reset in its dashboard): the seat follows.
            g.recenter_requested = true;
        }
        event = {XR_TYPE_EVENT_DATA_BUFFER};
    }
}

void Barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to, VkAccessFlags src_access,
             VkAccessFlags dst_access) {
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcAccessMask = src_access;
    barrier.dstAccessMask = dst_access;
    barrier.oldLayout = from;
    barrier.newLayout = to;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrier);
}

// Development (BB_VR_IPD, or the IPD trigger file): the eye distance in metres; 0 draws both eyes at the head, for
// ground truths (the head moved to an eye renders what that eye should see). Below 0: the runtime's.
std::atomic<float> g_ipd_override{[] {
    const char* v = std::getenv("BB_VR_IPD");
    return v ? float(std::atof(v)) : -1.f;
}()};

// The head in the LOCAL space at the predicted display time, and the union of the eyes' fields of view, for the camera.
void UpdateHead() {
    XrSpaceLocation location{XR_TYPE_SPACE_LOCATION};
    if (!XR_SUCCEEDED(g.xr.LocateSpace(g.view_space, g.local_space, g.frame_state.predictedDisplayTime, &location)) ||
        !(location.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT)) {
        return;
    }
    XrViewLocateInfo info{XR_TYPE_VIEW_LOCATE_INFO};
    info.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
    info.displayTime = g.frame_state.predictedDisplayTime;
    info.space = g.local_space;
    XrViewState state{XR_TYPE_VIEW_STATE};
    XrView views[2] = {{XR_TYPE_VIEW}, {XR_TYPE_VIEW}};
    u32 count = 0;
    if (!XR_SUCCEEDED(g.xr.LocateViews(g.session, &info, &state, 2, &count, views)) || count != 2) {
        return;
    }
    // Development: BB_VR_WOBBLE=<degrees> sways the tracked head's yaw by that much at 1 Hz (head motion on demand).
    static const float wobble = [] {
        const char* v = std::getenv("BB_VR_WOBBLE");
        return v ? float(std::atof(v)) : 0.f;
    }();
    if (wobble != 0.f) {
        const float t = float(g.frame_state.predictedDisplayTime % 1000000000) * 1e-9f;
        const float a = 0.5f * wobble * 0.0174533f * std::sin(6.28318f * t);
        const XrQuaternionf r{0.f, std::sin(a), 0.f, std::cos(a)}, o = location.pose.orientation;
        location.pose.orientation = {r.w * o.x + r.x * o.w + r.y * o.z - r.z * o.y,
                                     r.w * o.y - r.x * o.z + r.y * o.w + r.z * o.x,
                                     r.w * o.z + r.x * o.y - r.y * o.x + r.z * o.w,
                                     r.w * o.w - r.x * o.x - r.y * o.y - r.z * o.z};
    }
    const float layer_pose[7] = {location.pose.orientation.x, location.pose.orientation.y, location.pose.orientation.z,
                                 location.pose.orientation.w, location.pose.position.x,    location.pose.position.y,
                                 location.pose.position.z};
    {
        const XrQuaternionf& o = location.pose.orientation; // development (BB_VR_POSELOG): the head's yaw, degrees
        g.log_head_yaw = std::atan2(2.f * (o.w * o.y + o.x * o.z), 1.f - 2.f * (o.y * o.y + o.x * o.x)) * 57.2958f;
    }
    XrQuaternionf q = location.pose.orientation;
    XrVector3f p = location.pose.position;
    // Yaw of the head's forward (-Z) about +Y.
    const auto yaw_of = [](const XrQuaternionf& o) {
        const float fx = -2.f * (o.x * o.z + o.w * o.y), fz = -(1.f - 2.f * (o.x * o.x + o.y * o.y));
        return std::atan2(-fx, -fz);
    };
    if (!g.have_reference || g.recenter_requested) {
        g.reference_yaw = yaw_of(q);
        g.reference_position = p;
        g.have_reference = true;
        g.recenter_requested = false;
        Log("recentred (yaw %.1f degrees)", g.reference_yaw * 57.2958f);
    }
    {
        // Relative to the reference: undo its yaw (a rotation about +Y) and its position.
        const float h = -g.reference_yaw * 0.5f;
        const XrQuaternionf r{0.f, std::sin(h), 0.f, std::cos(h)};
        const XrQuaternionf o = q;
        q = {r.w * o.x + r.x * o.w + r.y * o.z - r.z * o.y, r.w * o.y - r.x * o.z + r.y * o.w + r.z * o.x,
             r.w * o.z + r.x * o.y - r.y * o.x + r.z * o.w, r.w * o.w - r.x * o.x - r.y * o.y - r.z * o.z};
        const float dx = p.x - g.reference_position.x, dy = p.y - g.reference_position.y,
                    dz = p.z - g.reference_position.z;
        const float c = std::cos(-g.reference_yaw), s = std::sin(-g.reference_yaw);
        p = {c * dx + s * dz, dy, -s * dx + c * dz};
    }
    // Rows: the images of the head's x, y and z axes, then its position (row vectors, as the game's matrices).
    const float xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z, xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z,
                wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
    const float pose[16] = {1 - 2 * (yy + zz), 2 * (xy + wz),     2 * (xz - wy),     0,
                            2 * (xy - wz),     1 - 2 * (xx + zz), 2 * (yz + wx),     0,
                            2 * (xz + wy),     2 * (yz - wx),     1 - 2 * (xx + yy), 0,
                            p.x,               p.y,               p.z,               1};
    const float l = std::tan(std::min(views[0].fov.angleLeft, views[1].fov.angleLeft));
    const float r = std::tan(std::max(views[0].fov.angleRight, views[1].fov.angleRight));
    const float u = std::tan(std::max(views[0].fov.angleUp, views[1].fov.angleUp));
    const float d = std::tan(std::min(views[0].fov.angleDown, views[1].fov.angleDown));
    Camera::SetHead(pose, l, r, u, d, layer_pose);
    const float ex = views[1].pose.position.x - views[0].pose.position.x,
                ey = views[1].pose.position.y - views[0].pose.position.y,
                ez = views[1].pose.position.z - views[0].pose.position.z;
    g.ipd = std::sqrt(ex * ex + ey * ey + ez * ez);
    if (const float ipd = g_ipd_override.load(std::memory_order_relaxed); ipd >= 0.f) {
        g.ipd = ipd;
    }
}


// Every 900 submitted frames: average rate, 1% low (the 99th percentile interval), late frames (an interval over 1.5
// display periods), and how many were repeats of the last stereo frame or the flat screen.
void NoteFrameTime() {
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    if (g.last_submit > 0.0) {
        g.intervals.push_back(t - g.last_submit);
    }
    g.last_submit = t;
    static const size_t report_frames = [] {
        const char* v = std::getenv("BB_VR_REPORT_FRAMES"); // development: frames per report (900: 10 s at 90 Hz)
        return v ? std::max<size_t>(90, size_t(std::atoi(v))) : size_t(900);
    }();
    if (g.intervals.size() < report_frames) {
        return;
    }
    const double period = g.frame_state.predictedDisplayPeriod > 0 ? double(g.frame_state.predictedDisplayPeriod) * 1e-9
                                                                      : 1.0 / 90.0;
    double sum = 0.0;
    u32 late = 0, over = 0; // over: past the display period (a missed refresh on a headset)
    for (const double i : g.intervals) {
        sum += i;
        late += i > 1.5 * period;
        over += i > 1.1 * period;
    }
    std::sort(g.intervals.begin(), g.intervals.end());
    const double p99 = g.intervals[g.intervals.size() * 99 / 100];
    // The game's own rate is the headset frames that showed a new game frame (the rest repeat the last one at the new
    // head pose): the headset holds its refresh rate whatever the game does.
    const double headset_fps = double(g.intervals.size()) / sum;
    const double new_frames = double(g.intervals.size() - std::min<size_t>(g.repeats, g.intervals.size()));
    Log("game %.1f FPS (new frames); headset %.1f FPS, 1%% low %.1f FPS, over a period %.2f%%, late %.2f%% (display "
        "%.1f Hz); %u repeated, %u flat screen",
        new_frames / sum, headset_fps, 1.0 / p99, 100.0 * over / double(g.intervals.size()),
        100.0 * late / double(g.intervals.size()), 1.0 / period, g.repeats, g.screens);
    {
        u32 renders = 0, captures = 0;
        Stereo::TakeCounts(renders, captures);
        Log("per second: game flips %.1f, game 3D frames %.1f, stereo frames taken %.1f, published %.1f, headset frames "
            "%.1f",
            g.count_flips.exchange(0) / sum, renders / sum, captures / sum, g.count_publications.exchange(0) / sum,
            double(g.intervals.size()) / sum);
    }
    const double n = double(g.intervals.size());
    Log("headset loop ms (avg / max): wait frame %.2f / %.1f, game frame %.2f / %.1f, begin %.2f / %.1f, show %.2f / "
        "%.1f, end frame %.2f / %.1f",
        g.stage_sum[0] / n, g.stage_max[0], g.stage_sum[4] / n, g.stage_max[4], g.stage_sum[1] / n, g.stage_max[1],
        g.stage_sum[2] / n, g.stage_max[2], g.stage_sum[3] / n, g.stage_max[3]);
    Log("show parts ms (sum over the report / max): acquire %.0f / %.1f, command buffer %.0f / %.1f, record %.0f / %.1f, "
        "submit %.0f / %.1f, release %.0f / %.1f", g.part_sum[0], g.part_max[0], g.part_sum[1], g.part_max[1],
        g.part_sum[2], g.part_max[2], g.part_sum[4], g.part_max[4], g.part_sum[3], g.part_max[3]);
    g.part_sum = {};
    g.part_max = {};
    g.stage_sum = {};
    g.stage_max = {};
    g.intervals.clear();
    g.repeats = g.screens = 0;
}

// The headset's thread: a command buffer of its own (the one used three frames ago has finished).
VkCommandBuffer NextCommand() {
    const u32 i = g.next_cmd;
    g.next_cmd = (i + 1) % 3;
    VULKAN_HPP_DEFAULT_DISPATCHER.vkWaitForFences(g.device, 1, &g.fences[i], VK_TRUE, UINT64_MAX);
    VULKAN_HPP_DEFAULT_DISPATCHER.vkResetFences(g.device, 1, &g.fences[i]);
    VULKAN_HPP_DEFAULT_DISPATCHER.vkResetCommandBuffer(g.cmds[i], 0);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VULKAN_HPP_DEFAULT_DISPATCHER.vkBeginCommandBuffer(g.cmds[i], &begin);
    return g.cmds[i];
}

// Submits `cmd` on the headset's queue once the publication's GPU work is done (its timeline semaphore's tick).
void SubmitCommand(VkCommandBuffer cmd, const State::Published& p) {
    VULKAN_HPP_DEFAULT_DISPATCHER.vkEndCommandBuffer(cmd);
    const u32 i = (g.next_cmd + 2) % 3; // the one NextCommand just handed out
    VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    timeline.waitSemaphoreValueCount = 1;
    timeline.pWaitSemaphoreValues = &p.tick;
    const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.pNext = &timeline;
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &p.semaphore;
    submit.pWaitDstStageMask = &stage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    QueueLock lock;
    VULKAN_HPP_DEFAULT_DISPATCHER.vkQueueSubmit(g.queue, 1, &submit, g.fences[i]);
}

void Release(XrSwapchain swapchain) {
    XrSwapchainImageReleaseInfo release{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
    QueueLock lock;
    g.xr.ReleaseSwapchainImage(swapchain, &release);
}

// The headset's thread: a new publication into the swapchains. False: nothing shown (the last content stays).
bool Show(const State::Published& p) {
    if (p.kind == State::Content::Stereo) {
        const auto& stereo = p.stereo;
        using Clock = std::chrono::steady_clock;
        auto since = Clock::now();
        const auto part = [&since](int i) {
            const auto now = Clock::now();
            const double ms = std::chrono::duration<double, std::milli>(now - since).count();
            g.part_sum[i] += ms;
            g.part_max[i] = std::max(g.part_max[i], ms);
            since = now;
        };
        const bool reticle = Stereo::ReticleDistance() > 0.f;
        if (!MakeChain(g.eye_chains[0], stereo.width, stereo.height, stereo.rgba) ||
            !MakeChain(g.eye_chains[1], stereo.width, stereo.height, stereo.rgba) ||
            !MakeChain(g.hud_chain, stereo.width, stereo.height, stereo.rgba) ||
            (reticle && !MakeChain(g.far_chain, stereo.width, stereo.height, stereo.rgba)) ||
            !AcquireChain(g.eye_chains[0]) || !AcquireChain(g.eye_chains[1]) || !AcquireChain(g.hud_chain) ||
            (reticle && !AcquireChain(g.far_chain))) {
            return false;
        }
        part(0);
        // Shift per unit of reversed depth: -side * (ipd / 2) * P00 * (width / 2) / near (vr_stereo.comp).
        const float p00 = 2.f / (p.tangents[1] - p.tangents[0]);
        const float base = 0.5f * StereoBaseline() * p00 * 0.5f * float(stereo.width) / Camera::Near();
        const float k[2] = {base, -base}; // left eye (side -1): surfaces move right
        const VkImage eyes[2] = {g.eye_chains[0].images[g.eye_chains[0].index].image,
                                 g.eye_chains[1].images[g.eye_chains[1].index].image};
        const VkCommandBuffer cmd = NextCommand();
        part(1);
        Stereo::RecordEyes(cmd, stereo, eyes, k, g.hud_chain.images[g.hud_chain.index].image,
                           reticle ? g.far_chain.images[g.far_chain.index].image : VK_NULL_HANDLE);
        part(2);
        SubmitCommand(cmd, p);
        part(4);
        Release(g.eye_chains[0].swapchain);
        Release(g.eye_chains[1].swapchain);
        Release(g.hud_chain.swapchain);
        if (reticle) {
            Release(g.far_chain.swapchain);
        }
        part(3);
        std::memcpy(g.frame_layer_pose, p.layer_pose, sizeof(g.frame_layer_pose));
        std::memcpy(g.frame_tangents, p.tangents, sizeof(g.frame_tangents));
        g.log_tagged = p.stereo.has_pose; // development (BB_VR_POSELOG)
        g.log_serial = p.serial;
        if (g.content == State::Content::Screen) {
            static const bool auto_recenter = !(std::getenv("BB_VR_AUTO_RECENTER") &&
                                                std::getenv("BB_VR_AUTO_RECENTER")[0] == '0');
            const double flat = std::chrono::duration<double>(std::chrono::steady_clock::now() - g.screen_since).count();
            if (auto_recenter && flat >= 2.0) {
                Log("stereo again after %.1f s of loading screen: recentring", flat);
                g.recenter_requested = true;
            }
        }
        g.content = State::Content::Stereo;
        return true;
    }
    if ((g.width != p.screen_width || g.height != p.screen_height || !g.swapchain) &&
        !CreateSwapchain(p.screen_width, p.screen_height)) {
        return false;
    }
    u32 index = 0;
    {
        QueueLock lock;
        XrSwapchainImageAcquireInfo acquire{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
        if (!Check(g.xr.AcquireSwapchainImage(g.swapchain, &acquire, &index), "xrAcquireSwapchainImage")) {
            return false;
        }
        XrSwapchainImageWaitInfo image_wait{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
        image_wait.timeout = XR_INFINITE_DURATION;
        g.xr.WaitSwapchainImage(g.swapchain, &image_wait);
    }
    const VkCommandBuffer cmd = NextCommand();
    const VkImage target = g.images[index].image;
    Barrier(cmd, target, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    VkImageCopy copy{};
    copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.extent = {g.width, g.height, 1};
    VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdCopyImage(cmd, g.screen_images[p.screen], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, target,
                                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    Stereo::FrameLog(cmd, g.screen_images[p.screen], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g.width, g.height); // dev
    Barrier(cmd, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT);
    SubmitCommand(cmd, p);
    Release(g.swapchain);
    if (g.content != State::Content::Screen) {
        g.screen_since = std::chrono::steady_clock::now();
    }
    g.content = State::Content::Screen;
    g.shown_movie = p.movie;
    return true;
}

void EndFrame() {
    // The screen: 3 m wide, 2.5 m ahead of where the head was when the session started, at eye height.
    XrCompositionLayerQuad quad{XR_TYPE_COMPOSITION_LAYER_QUAD};
    quad.space = g.local_space;
    quad.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
    quad.subImage.swapchain = g.swapchain;
    quad.subImage.imageRect = {{0, 0}, {s32(g.width), s32(g.height)}};
    quad.pose.orientation.w = 1.f;
    quad.pose.position = {0.f, 0.f, -2.5f};
    // A movie is fitted into the render's shape (Fixes::MovieCanvas): the screen at that shape. The menus are laid out
    // for 16:9 whatever the render's shape (Matt 2026-10-26: a round button tall in a square render): 16:9.
    quad.size = {3.f, 3.f / (g.shown_movie ? Camera::ScreenAspect() : 16.f / 9.f)};
    const XrCompositionLayerBaseHeader* layers[3] = {reinterpret_cast<XrCompositionLayerBaseHeader*>(&quad), nullptr,
                                                     nullptr};
    float layer_pose[7], tangents[4];
    XrCompositionLayerProjection projection{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
    XrCompositionLayerProjectionView views[2] = {{XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW},
                                                 {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW}};
    XrCompositionLayerQuad hud{XR_TYPE_COMPOSITION_LAYER_QUAD}, reticle_layer{XR_TYPE_COMPOSITION_LAYER_QUAD};
    u32 layer_count = g.content == State::Content::Screen ? 1 : 0;
    const bool stereo = g.content == State::Content::Stereo;
    if (stereo) {
        std::memcpy(layer_pose, g.frame_layer_pose, sizeof(layer_pose));
        std::memcpy(tangents, g.frame_tangents, sizeof(tangents));
    }
    if (stereo) {
        const XrQuaternionf q{layer_pose[0], layer_pose[1], layer_pose[2], layer_pose[3]};
        // The head's x axis (the first row of its rotation) for the eyes' positions.
        const float axis[3] = {1 - 2 * (q.y * q.y + q.z * q.z), 2 * (q.x * q.y + q.w * q.z), 2 * (q.x * q.z - q.w * q.y)};
        for (int eye = 0; eye < 2; ++eye) {
            auto& view = views[eye];
            const float side = (eye ? 0.5f : -0.5f) * g.ipd;
            view.pose.orientation = q;
            view.pose.position = {layer_pose[4] + side * axis[0], layer_pose[5] + side * axis[1],
                                  layer_pose[6] + side * axis[2]};
            view.fov = {std::atan(tangents[0]), std::atan(tangents[1]), std::atan(tangents[2]), std::atan(tangents[3])};
            view.subImage.swapchain = g.eye_chains[eye].swapchain;
            view.subImage.imageRect = {{0, 0}, {s32(g.eye_chains[eye].width), s32(g.eye_chains[eye].height)}};
        }
        projection.space = g.local_space;
        projection.viewCount = 2;
        projection.views = views;
        layers[0] = reinterpret_cast<XrCompositionLayerBaseHeader*>(&projection);
        layer_count = 1;
        {
            // The HUD: a screen 1.4 m wide, 1.1 m ahead of the seat (the recentred origin), level, over the scene.
            hud.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
            hud.space = g.local_space;
            hud.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
            hud.subImage.swapchain = g.hud_chain.swapchain;
            hud.subImage.imageRect = {{0, 0}, {s32(g.hud_chain.width), s32(g.hud_chain.height)}};
            const float yaw = g.reference_yaw;
            hud.pose.orientation = {0.f, std::sin(yaw * 0.5f), 0.f, std::cos(yaw * 0.5f)};
            hud.pose.position = {g.reference_position.x - 1.1f * std::sin(yaw), g.reference_position.y,
                                 g.reference_position.z - 1.1f * std::cos(yaw)};
            // The game lays its HUD out for 16:9 whatever the render's shape (a square eye size: the reticle a tall
            // ellipse in the canvas; Matt 2026-10-28: "the in game HUD is stretched vertically"), as the menus do.
            hud.size = {1.4f, 1.4f * 9.f / 16.f};
            layers[1] = reinterpret_cast<XrCompositionLayerBaseHeader*>(&hud);
            layer_count = 2;
            const float reticle_distance = Stereo::ReticleDistance();
            if (reticle_distance > 0.f && g.far_chain.swapchain) {
                // The reticle layer: the HUD's centre box on its own quad, reticle_distance ahead, at the same angular
                // size (so aiming converges in the scene; the HUD holds transparent black there).
                u32 x, y, bw, bh;
                Stereo::ReticleBox(g.far_chain.width, g.far_chain.height, x, y, bw, bh);
                reticle_layer = hud;
                reticle_layer.subImage.swapchain = g.far_chain.swapchain;
                reticle_layer.subImage.imageRect = {{s32(x), s32(y)}, {s32(bw), s32(bh)}};
                const float scale = reticle_distance / 1.1f;
                reticle_layer.size = {hud.size.width * scale * float(bw) / float(g.far_chain.width),
                            hud.size.height * scale * float(bh) / float(g.far_chain.height)};
                reticle_layer.pose.position = {g.reference_position.x - reticle_distance * std::sin(yaw), g.reference_position.y,
                                     g.reference_position.z - reticle_distance * std::cos(yaw)};
                layers[2] = reinterpret_cast<XrCompositionLayerBaseHeader*>(&reticle_layer);
                layer_count = 3;
            }
        }
    }
    XrFrameEndInfo end{XR_TYPE_FRAME_END_INFO};
    end.displayTime = g.frame_state.predictedDisplayTime;
    end.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
    end.layerCount = layer_count;
    end.layers = layers;
    static int pose_log = [] {
        const char* v = std::getenv("BB_VR_POSELOG");
        return v ? std::atoi(v) : 0;
    }();
    if (pose_log > 0) { // development: one line per headset frame (see State::log_head_yaw)
        static FILE* f = [] {
            const char* temp = std::getenv("TEMP");
            return std::fopen((std::string(temp ? temp : ".") + "/vr_poselog.txt").c_str(), "w");
        }();
        if (f) {
            const float* l = g.frame_layer_pose;
            const float sub = std::atan2(2.f * (l[3] * l[1] + l[0] * l[2]), 1.f - 2.f * (l[1] * l[1] + l[0] * l[0])) *
                              57.2958f;
            std::fprintf(f, "%lld %.3f %.3f %d %d %d %llu %d\n", (long long)g.frame_state.predictedDisplayTime,
                         g.log_head_yaw, sub, int(stereo), int(g.log_new_frame), int(g.log_tagged),
                         (unsigned long long)g.log_serial, int(g.frame_state.predictedDisplayPeriod / 1000));
            if (--pose_log == 0) {
                std::fclose(f);
                f = nullptr;
                Log("pose log written");
            }
        }
    }
    QueueLock lock;
    Check(g.xr.EndFrame(g.session, &end), "xrEndFrame");
}

// The headset's frame loop: paced by the runtime, independent of the game's presentation.
// A publication whose GPU work is done (its timeline tick reached). One that is still being drawn waits for a later
// headset frame (the last one is shown again meanwhile): its copies would wait on the game's GPU work and the runtime's
// compositor on them, and the headset's frame would miss its refresh (GPU-bound at large eye sizes: ~10% of frames).
// BB_VR_WAIT_FRAMES=1: the old way (development A/B).
bool Finished(const State::Published& p) {
    static const bool wait = std::getenv("BB_VR_WAIT_FRAMES") && std::getenv("BB_VR_WAIT_FRAMES")[0] == '1';
    if (wait || !p.semaphore) {
        return true;
    }
    u64 value = 0;
    return VULKAN_HPP_DEFAULT_DISPATCHER.vkGetSemaphoreCounterValue(g.device, p.semaphore, &value) != VK_SUCCESS ||
           value >= p.tick;
}

// The headset's thread, after xrWaitFrame: waits (up to 100 ms) for the game's next publication and its GPU work, so a
// slow game frame makes the headset frame late and the runtime sees the drop (SteamVR's frame timing, its reprojection
// and motion smoothing). Before, the loop submitted the last frame again on time: SteamVR always reported the full
// rate. The 100 ms limit keeps the headset fed through loading screens. BB_VR_REPEAT=1: the old way.
void WaitForGameFrame() {
    static const bool repeat = std::getenv("BB_VR_REPEAT") && std::getenv("BB_VR_REPEAT")[0] == '1';
    if (repeat) {
        return;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    State::Published newest;
    {
        std::unique_lock lock{g.publish_mutex};
        if (!g.publish_cv.wait_until(lock, deadline, [] {
                return g.quit.load() || (g.recent[0].kind != State::Content::None && g.recent[0].serial > g.shown_serial);
            })) {
            return;
        }
        newest = g.recent[0];
    }
    const auto left = std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - std::chrono::steady_clock::now());
    if (!newest.semaphore || left.count() <= 0) {
        return;
    }
    VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    info.semaphoreCount = 1;
    info.pSemaphores = &newest.semaphore;
    info.pValues = &newest.tick;
    VULKAN_HPP_DEFAULT_DISPATCHER.vkWaitSemaphores(g.device, &info, u64(left.count()));
}

void FrameLoop() {
    while (!g.quit) {
        PollEvents();
        if (!g.running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        g.frame_state = {XR_TYPE_FRAME_STATE};
        XrFrameWaitInfo wait{XR_TYPE_FRAME_WAIT_INFO};
        using Clock = std::chrono::steady_clock;
        const auto stage = [](int i, Clock::time_point& since) {
            const auto now = Clock::now();
            const double ms = std::chrono::duration<double, std::milli>(now - since).count();
            g.stage_sum[i] += ms;
            g.stage_max[i] = std::max(g.stage_max[i], ms);
            since = now;
        };
        auto since = Clock::now();
        if (!Check(g.xr.WaitFrame(g.session, &wait, &g.frame_state), "xrWaitFrame")) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        stage(0, since);
        if (g.frame_state.predictedDisplayPeriod > 0) {
            const u64 period = u64(g.frame_state.predictedDisplayPeriod);
            g.display_period_ns.store(period, std::memory_order_relaxed);
            if (!g.xr.GetDisplayRefreshRateFB) { // no refresh rate from the runtime: the shortest period seen
                const u64 shortest = g.refresh_period_ns.load(std::memory_order_relaxed);
                if (!shortest || period < shortest) {
                    g.refresh_period_ns.store(period, std::memory_order_relaxed);
                }
            }
        }
        {
            std::scoped_lock lock{g_flip_token_mutex};
            g.flip_token.store(true, std::memory_order_relaxed); // one game flip per headset frame (while capped)
        }
        g_flip_token_cv.notify_all();
        // VR fork: the head for this frame's display time before the wait, so the game's next frame renders with it.
        UpdateHead();
        if (float pose[7], tangents[4]; Multiview::Enabled() && Camera::LastRendered(pose, tangents)) {
            // An eye half the eye distance to the side moves clip x by P00 x that distance (left eye: surfaces right).
            Multiview::SetShift(2.f / (tangents[1] - tangents[0]) * 0.5f * StereoBaseline());
        }
        WaitForGameFrame(); // VR fork: late when the game is (the runtime sees the dropped frame)
        stage(4, since);
        std::scoped_lock frame_lock{g.frame_mutex};
        {
            QueueLock lock;
            XrFrameBeginInfo begin{XR_TYPE_FRAME_BEGIN_INFO};
            if (!Check(g.xr.BeginFrame(g.session, &begin), "xrBeginFrame")) {
                continue;
            }
        }
        stage(1, since);
        std::array<State::Published, 3> recent;
        {
            std::scoped_lock lock{g.publish_mutex};
            recent = g.recent;
        }
        bool shown = false;
        for (size_t i = 0; i < recent.size() && g.frame_state.shouldRender; ++i) {
            const State::Published& p = recent[i];
            if (p.kind == State::Content::None || p.serial <= g.shown_serial ||
                (p.kind == State::Content::Screen && i >= 2)) {
                break; // older ones were shown, or their images may be refilled
            }
            if (Finished(p)) {
                g.shown_serial = p.serial;
                shown = Show(p);
                break;
            }
        }
        g.repeats += !shown;
        g.log_new_frame = shown; // development (BB_VR_POSELOG)
        g.screens += shown && g.content == State::Content::Screen;
        stage(2, since);
        EndFrame();
        stage(3, since);
        NoteFrameTime();
    }
}
} // namespace

bool Active() {
    return g.instance != XR_NULL_HANDLE;
}

u32 QueueCount(VkPhysicalDevice physical_device, u32 queue_family, u32 used) {
    g.queue_count = used;
    if (!Active() || (std::getenv("BB_VR_SHARED_QUEUE") && std::getenv("BB_VR_SHARED_QUEUE")[0] == '1')) {
        return used;
    }
    u32 count = 0;
    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, families.data());
    if (queue_family < count && families[queue_family].queueCount > used) {
        g.queue_index = used;
        g.queue_count = used + 1;
    }
    return g.queue_count;
}

const std::vector<const char*>& InstanceExtensions() {
    std::call_once(g.init_once, Init);
    return g.instance_extensions;
}

const std::vector<const char*>& DeviceExtensions(VkInstance instance, VkPhysicalDevice physical_device) {
    if (!Active() || g.device_extensions_queried) {
        return g.device_extensions;
    }
    g.device_extensions_queried = true;
    VkPhysicalDevice wanted{};
    if (Check(g.xr.GetVulkanGraphicsDeviceKHR(g.instance, g.system, instance, &wanted), "xrGetVulkanGraphicsDeviceKHR") &&
        wanted != physical_device) {
        Log("warning: the headset's GPU is not the one the renderer chose");
    }
    u32 size = 0;
    g.xr.GetVulkanDeviceExtensionsKHR(g.instance, g.system, 0, &size, nullptr);
    g.device_extension_text.resize(size);
    g.xr.GetVulkanDeviceExtensionsKHR(g.instance, g.system, size, &size, g.device_extension_text.data());
    g.device_extension_text.resize(size ? size - 1 : 0);
    g.device_extensions = Split(g.device_extension_text);
    Log("Vulkan device extensions: %s", g.device_extension_text.c_str());
    return g.device_extensions;
}

void OnDeviceCreated(VkInstance instance, VkPhysicalDevice physical_device, VkDevice device, u32 queue_family) {
    if (!Active()) {
        return;
    }
    XrGraphicsRequirementsVulkanKHR requirements{XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN_KHR};
    g.xr.GetVulkanGraphicsRequirementsKHR(g.instance, g.system, &requirements);
    XrGraphicsBindingVulkanKHR binding{XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR};
    binding.instance = instance;
    binding.physicalDevice = physical_device;
    binding.device = device;
    binding.queueFamilyIndex = queue_family;
    g.own_queue = g.queue_index != 0;
    binding.queueIndex = g.queue_index; // the headset's own queue when the device was made with two
    XrSessionCreateInfo info{XR_TYPE_SESSION_CREATE_INFO};
    info.next = &binding;
    info.systemId = g.system;
    {
        QueueLock lock;
        if (!Check(g.xr.CreateSession(g.instance, &info, &g.session), "xrCreateSession")) {
            g.session = XR_NULL_HANDLE;
            return;
        }
    }
    g.device = device;
    g.physical_device = physical_device;
    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceQueue(device, queue_family, g.queue_index, &g.queue);
    {
        VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = queue_family;
        VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateCommandPool(device, &pool_info, nullptr, &g.pool);
        VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        alloc.commandPool = g.pool;
        alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc.commandBufferCount = 3;
        VULKAN_HPP_DEFAULT_DISPATCHER.vkAllocateCommandBuffers(device, &alloc, g.cmds);
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        for (VkFence& fence : g.fences) {
            VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateFence(device, &fence_info, nullptr, &fence);
        }
    }
    {
        VkPhysicalDeviceVulkan11Features vk11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features.pNext = &vk11;
        VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceFeatures2(physical_device, &features);
        Multiview::OnDevice(device, vk11.multiview == VK_TRUE);
    }
    XrReferenceSpaceCreateInfo space{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
    space.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
    space.poseInReferenceSpace.orientation.w = 1.f;
    Check(g.xr.CreateReferenceSpace(g.session, &space, &g.local_space), "xrCreateReferenceSpace");
    space.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
    Check(g.xr.CreateReferenceSpace(g.session, &space, &g.view_space), "xrCreateReferenceSpace (view)");
    PickFormat();
    Log("session created (%s queue)", g.own_queue ? "its own" : "the game's");
    Stereo::Init(physical_device, device);
    g.thread = std::thread{FrameLoop};
}

void RecordFrame(vk::CommandBuffer cmd, vk::Image frame, u32 width, u32 height, vk::Image window, u32 window_width,
                 u32 window_height) {
    Dev::Poll();
#ifdef _WIN32
    // F8: the seat is where the head is now (also SteamVR's "reset seated position").
    static bool f8_down = false;
    const bool f8 = (GetAsyncKeyState(VK_F8) & 0x8000) != 0;
    if (f8 && !f8_down) {
        Recenter();
    }
    f8_down = f8;
#endif
    g.pending = {};
    ++g.count_flips; // development: the VR report's game flips
    if (!g.session) {
        return;
    }
    Camera::Install();
    // Uncharted fork: a movie goes to the screen at once (the whole frame: Fixes::MovieCanvas).
    const u32 source_width = width, source_height = height;
    u32 movie_width = 0, movie_height = 0;
    const bool movie = Fixes::MovieRect(movie_width, movie_height);
    // A stereo frame: published with the pose it was rendered with; the desktop window shows it.
    Stereo::Frame stereo{};
    float layer_pose[7], tangents[4];
    if (Stereo::TakeFrame(stereo) && !movie && (stereo.has_pose || Camera::LastRendered(layer_pose, tangents)) &&
        Camera::Near() > 0.f) {
        auto& p = g.pending;
        p.kind = State::Content::Stereo;
        p.stereo = stereo;
        std::memcpy(p.layer_pose, stereo.has_pose ? stereo.layer_pose : layer_pose, sizeof(p.layer_pose));
        std::memcpy(p.tangents, stereo.has_pose ? stereo.tangents : tangents, sizeof(p.tangents));
        Stereo::RecordMirror(cmd, stereo, VkImage(window), window_width, window_height);
        g.frames_since_stereo = 0;
        return;
    }
    // A late frame while racing: the headset keeps the last eyes. Menus without 3D go to the screen.
    if (g.frames_since_stereo < 45 && !movie) {
        ++g.frames_since_stereo;
        return;
    }
    g.frames_since_stereo = 45;
    if (g.format == VK_FORMAT_UNDEFINED || !MakeScreens(source_width, source_height)) {
        return;
    }
    const u32 index = g.screen_next;
    g.screen_next = (index + 1) % 3;
    const VkCommandBuffer c = cmd;
    const VkImage target = g.screen_images[index];
    Barrier(c, target, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[1] = {s32(source_width), s32(source_height), 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstOffsets[1] = {s32(source_width), s32(source_height), 1};
    VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdBlitImage(c, VkImage(frame), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, target,
                                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
    Barrier(c, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    g.pending.kind = State::Content::Screen;
    g.pending.screen = index;
    g.pending.screen_width = source_width;
    g.pending.screen_height = source_height;
    g.pending.movie = movie;
}

void SetIpdOverride(float ipd) {
    g_ipd_override.store(ipd, std::memory_order_relaxed);
}

float StereoBaseline() {
    static const float world_scale = [] {
        const char* v = std::getenv("BB_VR_WORLD_SCALE");
        return v ? float(std::atof(v)) : 1.f;
    }();
    return g.ipd * world_scale;
}

void Recenter() {
    g.recenter_requested = true;
}


std::chrono::nanoseconds FramePeriod(std::chrono::nanoseconds limit) {
    if (g.capped.load(std::memory_order_relaxed)) {
        g.flip_token.store(false, std::memory_order_relaxed);
    }
    return limit;
}

bool FlipAllowed() {
    return !g.capped.load(std::memory_order_relaxed) || g.flip_token.load(std::memory_order_relaxed);
}

bool ImmediateFlips() {
    static const bool on = [] {
        const char* v = std::getenv("BB_VR_IMMEDIATE_FLIPS");
        return !(v && v[0] == '0');
    }();
    return on && Active();
}

bool WaitFlipAllowed(std::chrono::steady_clock::time_point deadline) {
    std::unique_lock lock{g_flip_token_mutex};
    return g_flip_token_cv.wait_until(lock, deadline, [] { return FlipAllowed(); });
}

// The presenter, each game flip in VR: the adaptive cap's flip rate over 2 s windows.
void NoteFlip() {
    static const bool enabled = !(std::getenv("BB_VR_CAP") && std::getenv("BB_VR_CAP")[0] == '0');
    const u64 refresh = g.refresh_period_ns.load(std::memory_order_relaxed);
    const u64 period = refresh ? refresh : g.display_period_ns.load(std::memory_order_relaxed);
    if (!enabled || !period) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (g.flips_since == std::chrono::steady_clock::time_point{}) {
        g.flips_since = now;
    }
    ++g.flips;
    const double seconds = std::chrono::duration<double>(now - g.flips_since).count();
    if (seconds < 2.0) {
        return;
    }
    const double rate = g.flips / seconds, hz = 1e9 / double(period);
    const bool capped = g.capped.load(std::memory_order_relaxed);
    if (!capped && rate >= 1.12 * hz) {
        g.capped = true;
        Log("the game holds %.0f FPS: capped at the headset's %.0f Hz", rate, hz);
    } else if (capped && rate < 0.95 * hz) {
        g.capped = false;
        Log("the game makes %.0f FPS of the headset's %.0f Hz: uncapped", rate, hz);
    }
    g.flips = 0;
    g.flips_since = now;
}

void SubmitFrame(VkSemaphore semaphore, u64 tick) {
    if (g.pending.kind == State::Content::Stereo) {
        NoteFlip();
        ++g.count_publications;
    }
    if (g.pending.kind == State::Content::None) {
        return;
    }
    std::scoped_lock lock{g.publish_mutex};
    const u64 serial = g.published.serial + 1;
    g.published = g.pending;
    g.published.semaphore = semaphore;
    g.published.tick = tick;
    g.published.serial = serial;
    for (size_t i = g.recent.size() - 1; i > 0; --i) {
        g.recent[i] = g.recent[i - 1];
    }
    g.recent[0] = g.published;
    g.pending = {};
    g.publish_cv.notify_all(); // VR fork: the headset's WaitForGameFrame
}

} // namespace Vulkan::Vr
