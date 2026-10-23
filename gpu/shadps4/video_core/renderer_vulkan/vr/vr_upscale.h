// SPDX-License-Identifier: GPL-2.0-or-later
// VR (Uncharted fork): the port's temporal upscaler (vk_temporal_upscaler.cpp) in stereo. Its pass runs once per eye:
// the views it makes of the scene colour and depth show that eye's layer (multiview's two-layer images), DLSS
// evaluates the eye's own feature (each eye keeps its own history: bridge ABI 3), and its result goes back into that
// eye's layer. Native size only (DLAA): a reduced render would need the scene proxies in stereo.

#pragma once

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_dlss.h"
#include "video_core/renderer_vulkan/vr/vr_multiview.h"

namespace Vulkan::Vr::Upscale {

inline u32 g_layer = 0;

/// The eye whose layer the upscaler's views show (0 outside VR).
inline u32 Layer() {
    return g_layer;
}

/// Runs the upscaler's pass for each eye (once outside multiview stereo), each with its DLSS feature. In stereo only
/// DLSS runs (`dlss`): FSR and TAA keep one history, which the eyes would share (DLSS unavailable: no pass at all).
template <typename Run>
void ForEachEye(bool dlss, Run&& run) {
    const u32 eyes = Multiview::Enabled() ? 2 : 1;
    if (eyes > 1 && !dlss) {
        return;
    }
    for (u32 eye = 0; eye < eyes; ++eye) {
        g_layer = eye;
        if (Dlss* dlss = Dlss::Get(); dlss && eyes > 1) {
            dlss->SelectFeature(eye);
        }
        run();
    }
    g_layer = 0;
    if (Dlss* dlss = Dlss::Get(); dlss && eyes > 1) {
        dlss->SelectFeature(0);
    }
}

} // namespace Vulkan::Vr::Upscale
