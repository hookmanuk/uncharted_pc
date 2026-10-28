// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <unordered_map>
#include <utility>
#include <boost/container_hash/hash.hpp>

#include "common/types.h"
#include "video_core/amdgpu/tiling.h"
#include "video_core/buffer_cache/buffer.h"

namespace Vulkan {
struct Runtime;
}

namespace VideoCore {

struct ImageInfo;
struct Image;

class TileManager {
    struct TilingKey {
        AmdGpu::TileMode tile_mode;
        u32 num_bits;
        u32 num_samples;
        bool is_tiler;
        bool from_image = false;

        bool operator==(const TilingKey&) const = default;

        struct Hash {
            size_t operator()(const TilingKey& key) const {
                size_t hash = 0;
                boost::hash_combine(hash, key.tile_mode);
                boost::hash_combine(hash, key.num_bits);
                boost::hash_combine(hash, key.num_samples);
                boost::hash_combine(hash, key.is_tiler);
                boost::hash_combine(hash, key.from_image);
                return hash;
            }
        };
    };

public:
    explicit TileManager(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         Vulkan::Runtime& runtime, StreamBuffer& stream_buffer);
    ~TileManager();

    void TileImage(Image& in_image, std::span<vk::BufferImageCopy> buffer_copies,
                   const VideoCore::Buffer* out_buffer, u64 out_offset);

    std::pair<const Buffer*, u64> DetileImage(const VideoCore::Buffer* in_buffer, u64 in_offset,
                                              const ImageInfo& info);

private:
    void RecordTilingDispatch(vk::Pipeline pipeline, const vk::DescriptorBufferInfo& tiled,
                              const vk::DescriptorBufferInfo& linear,
                              const vk::DescriptorBufferInfo& params, u32 dim_x);

    vk::Pipeline GetTilingPipeline(const ImageInfo& info, bool is_tiler, bool from_image = false);

    /// bbport VR: a single-level 32 bpp depth image tiled straight into the buffer (no staging
    /// copy); false when the image is not one.
    bool TileDepthFromImage(Image& image, const VideoCore::Buffer* out_buffer, u64 out_offset,
                            const vk::DescriptorBufferInfo& params);

private:
    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    Vulkan::Runtime& runtime;
    StreamBuffer& stream_buffer;
    vk::UniqueDescriptorSetLayout desc_layout;
    vk::UniquePipelineLayout pl_layout;
    vk::UniqueDescriptorSetLayout image_desc_layout; ///< bbport VR: binding 1 a sampled image
    vk::UniquePipelineLayout image_pl_layout;
    vk::UniqueSampler image_sampler;
    std::unordered_map<TilingKey, vk::UniquePipeline, TilingKey::Hash> tiling_pipelines;
};

} // namespace VideoCore
