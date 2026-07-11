// Screen-path pipeline cache + descriptor machinery. RenderPassDescriptor is
// the pipeline key (doc 4.3): BuildRenderPassDescriptor stays the single
// decode point, this file only translates its typed fields into Vulkan state.
#include "EngineVulkan.hpp"
#include "TextureVulkan.hpp"

#include <Poseidon/Foundation/Logging/Logging.hpp>
#include <Poseidon/Graphics/Rendering/BuildRenderPassDescriptor.hpp>
#include <Poseidon/Graphics/Rendering/Lighting/Lights.hpp>
#include <Poseidon/World/Scene/Scene.hpp>

namespace Poseidon
{

// ── Frame-shared GPU resources ──────────────────────────────────────────────

bool EngineVulkan::InitPipelineResources()
{
    try
    {
        _uboAlign = (uint32_t)std::max<vk::DeviceSize>(_vk.deviceProps.limits.minUniformBufferOffsetAlignment, 16);

        // Rings. Vertex/index capacities are multiples of the GL33 window
        // sizes; UBO ring sized for ~16k constant snapshots per frame (every
        // soup flush AND every TL mesh draw takes one VS+PS slice pair).
        constexpr vk::DeviceSize kVertexRingBytes = 256 * 1024 * sizeof(TLVertex);
        constexpr vk::DeviceSize kIndexRingBytes = 512 * 1024 * sizeof(WORD);
        const vk::DeviceSize uboSlice = ((kVSConstFloats + kPSConstFloats) * sizeof(float) + 2 * _uboAlign);
        const vk::DeviceSize kUboRingBytes = 16384 * uboSlice;
        constexpr vk::DeviceSize kDynMeshRingBytes = 256 * 1024 * sizeof(SVertexVulkan);
        for (int i = 0; i < kFramesInFlight; ++i)
        {
            if (!_vertexBuffer[i].Create(_vk, kVertexRingBytes, vk::BufferUsageFlagBits::eVertexBuffer, "soup-vb") ||
                !_indexBuffer[i].Create(_vk, kIndexRingBytes, vk::BufferUsageFlagBits::eIndexBuffer, "soup-ib") ||
                !_uboBuffer[i].Create(_vk, kUboRingBytes, vk::BufferUsageFlagBits::eUniformBuffer, "const-ubo") ||
                !_dynMeshBuffer[i].Create(_vk, kDynMeshRingBytes, vk::BufferUsageFlagBits::eVertexBuffer, "dyn-mesh-vb"))
                return false;
        }

        // Samplers: filter(linear/point) x clampU x clampV — mirror of GL33's
        // 8 sampler objects. Index = point*4 + clampU*2 + clampV. Linear
        // samplers get 16x anisotropy like GL33's CreateSamplerStates —
        // without it terrain at grazing angles turns to mush.
        const bool anisoSupported = _vk.physicalDevice.getFeatures().samplerAnisotropy;
        const float maxAniso = std::min(16.0f, _vk.deviceProps.limits.maxSamplerAnisotropy);
        for (int i = 0; i < kSamplerCombos; ++i)
        {
            const bool point = (i & 4) != 0;
            const vk::Filter filter = point ? vk::Filter::eNearest : vk::Filter::eLinear;
            const vk::SamplerAddressMode addrU =
                (i & 2) ? vk::SamplerAddressMode::eClampToEdge : vk::SamplerAddressMode::eRepeat;
            const vk::SamplerAddressMode addrV =
                (i & 1) ? vk::SamplerAddressMode::eClampToEdge : vk::SamplerAddressMode::eRepeat;
            vk::SamplerCreateInfo info({}, filter, filter,
                                       point ? vk::SamplerMipmapMode::eNearest : vk::SamplerMipmapMode::eLinear, addrU,
                                       addrV, vk::SamplerAddressMode::eRepeat);
            info.maxLod = VK_LOD_CLAMP_NONE;
            if (!point && anisoSupported && maxAniso > 1.0f)
            {
                info.anisotropyEnable = true;
                info.maxAnisotropy = maxAniso;
            }
            _samplers[i] = _vk.device.createSampler(info);
        }

        // 1x1 white texture — bound for untextured draws so they render as
        // (vertex color x white).
        {
            vk::ImageCreateInfo imgInfo({}, vk::ImageType::e2D, vk::Format::eR8G8B8A8Unorm, vk::Extent3D(1, 1, 1), 1,
                                        1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
                                        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst);
            VkImageCreateInfo rawImgInfo = imgInfo;
            VmaAllocationCreateInfo allocInfo{};
            allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
            VkImage rawImage = VK_NULL_HANDLE;
            if (vmaCreateImage(_vk.allocator, &rawImgInfo, &allocInfo, &rawImage, &_whiteAlloc, nullptr) != VK_SUCCESS)
            {
                LOG_ERROR(Graphics, "VK: white texture allocation failed");
                return false;
            }
            _whiteImage = rawImage;
            _whiteView = _vk.device.createImageView(
                {{}, _whiteImage, vk::ImageViewType::e2D, vk::Format::eR8G8B8A8Unorm, {},
                 vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)});

            // One-time upload through a transient staging buffer.
            VulkanBuffer staging;
            if (!staging.Create(_vk, 4, vk::BufferUsageFlagBits::eTransferSrc, "white-staging"))
                return false;
            const uint32_t white = 0xFFFFFFFFu;
            memcpy(staging.mapped, &white, 4);

            vk::CommandPool pool = _vk.device.createCommandPool(
                vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eTransient, _vk.graphicsQueueFamily));
            vk::CommandBuffer cmd =
                _vk.device.allocateCommandBuffers({pool, vk::CommandBufferLevel::ePrimary, 1}).front();
            cmd.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

            vk::ImageMemoryBarrier2 toDst(vk::PipelineStageFlagBits2::eNone, {}, vk::PipelineStageFlagBits2::eCopy,
                                          vk::AccessFlagBits2::eTransferWrite, vk::ImageLayout::eUndefined,
                                          vk::ImageLayout::eTransferDstOptimal, vk::QueueFamilyIgnored,
                                          vk::QueueFamilyIgnored, _whiteImage,
                                          vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
            cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, toDst));

            vk::BufferImageCopy region(0, 0, 0, vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),
                                       {0, 0, 0}, {1, 1, 1});
            cmd.copyBufferToImage(staging.buffer, _whiteImage, vk::ImageLayout::eTransferDstOptimal, region);

            vk::ImageMemoryBarrier2 toRead(vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite,
                                           vk::PipelineStageFlagBits2::eFragmentShader,
                                           vk::AccessFlagBits2::eShaderSampledRead,
                                           vk::ImageLayout::eTransferDstOptimal,
                                           vk::ImageLayout::eShaderReadOnlyOptimal, vk::QueueFamilyIgnored,
                                           vk::QueueFamilyIgnored, _whiteImage,
                                           vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
            cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, toRead));

            cmd.end();
            vk::CommandBufferSubmitInfo cmdInfo(cmd);
            _vk.graphicsQueue.submit2(vk::SubmitInfo2({}, {}, cmdInfo), nullptr);
            _vk.graphicsQueue.waitIdle();
            _vk.device.destroyCommandPool(pool);
            staging.Destroy(_vk);
        }

        // Descriptor set layout: 2 dynamic UBOs + tex0 + tex1 (detail slot;
        // white fallback when the draw is single-textured) + the instanced-run
        // world-matrix array (dynamic offset selects the run's 16KB slice) +
        // the cascade shadow depth array (1x1 fallback until a depth pass runs).
        const vk::DescriptorSetLayoutBinding bindings[] = {
            {0, vk::DescriptorType::eUniformBufferDynamic, 1, vk::ShaderStageFlagBits::eVertex},
            {1, vk::DescriptorType::eUniformBufferDynamic, 1, vk::ShaderStageFlagBits::eFragment},
            {2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment},
            {3, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment},
            {4, vk::DescriptorType::eUniformBufferDynamic, 1, vk::ShaderStageFlagBits::eVertex},
            {5, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment},
        };
        _setLayout = _vk.device.createDescriptorSetLayout({{}, 6, bindings});
        // Per-draw world matrix + instanced flag ride as push constants
        // (VSTransform/VSShadow): mat4 at 0, vec4 flags at 64.
        const vk::PushConstantRange worldRange(vk::ShaderStageFlagBits::eVertex, 0, 80);
        _pipelineLayout = _vk.device.createPipelineLayout({{}, 1, &_setLayout, 1, &worldRange});

        // Per-frame descriptor pools: every flush/TL draw allocates a fresh
        // set, the pool is reset in InitDraw when the slot's fence proves the
        // GPU is done (doc 4.5). Sized for soup flushes + per-draw TL sets.
        constexpr uint32_t kMaxSetsPerFrame = 16384;
        const vk::DescriptorPoolSize poolSizes[] = {
            {vk::DescriptorType::eUniformBufferDynamic, 3 * kMaxSetsPerFrame},
            {vk::DescriptorType::eCombinedImageSampler, 3 * kMaxSetsPerFrame},
        };
        for (int f = 0; f < kFramesInFlight; ++f)
            _frameDescPool[f] = _vk.device.createDescriptorPool({{}, kMaxSetsPerFrame, 2, poolSizes});

        // Texture-upload staging arenas (mip streaming; overflow falls back to
        // dedicated one-shot buffers, see EngineVulkan_Upload.cpp).
        constexpr vk::DeviceSize kStagingArenaBytes = 24 * 1024 * 1024;
        for (int f = 0; f < kFramesInFlight; ++f)
        {
            if (!_stagingBuffer[f].Create(_vk, kStagingArenaBytes, vk::BufferUsageFlagBits::eTransferSrc,
                                          "tex-staging"))
                return false;
        }

        // Shadow-map sampler + 1x1 fallback array: binding 5 is statically
        // used by the lit shaders, so it must reference a valid depth-array
        // view even before (or without) any shadow depth pass.
        {
            vk::SamplerCreateInfo shadowInfo({}, vk::Filter::eNearest, vk::Filter::eNearest,
                                             vk::SamplerMipmapMode::eNearest, vk::SamplerAddressMode::eClampToEdge,
                                             vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge);
            _shadowSampler = _vk.device.createSampler(shadowInfo);
        }
        if (!CreateShadowFallbackTexture())
            return false;

        return CreateDepthTarget() && InitShaderModules();
    }
    catch (const std::exception& e)
    {
        LOG_ERROR(Graphics, "VK: pipeline resource init failed: {}", e.what());
        return false;
    }
}

void EngineVulkan::DestroyPipelineResources()
{
    for (auto& [key, pipeline] : _pipelines)
        _vk.device.destroyPipeline(pipeline);
    _pipelines.clear();
    DestroyShaderModules();
    DestroyDepthTarget();
    DestroyShadowResources();
    for (auto& pool : _frameDescPool)
    {
        if (pool)
            _vk.device.destroyDescriptorPool(pool);
        pool = nullptr;
    }
    if (_pipelineLayout)
        _vk.device.destroyPipelineLayout(_pipelineLayout);
    _pipelineLayout = nullptr;
    if (_setLayout)
        _vk.device.destroyDescriptorSetLayout(_setLayout);
    _setLayout = nullptr;
    if (_whiteView)
        _vk.device.destroyImageView(_whiteView);
    _whiteView = nullptr;
    if (_whiteImage)
        vmaDestroyImage(_vk.allocator, _whiteImage, _whiteAlloc);
    _whiteImage = nullptr;
    _whiteAlloc = nullptr;
    for (auto& s : _samplers)
    {
        if (s)
            _vk.device.destroySampler(s);
        s = nullptr;
    }
    for (int i = 0; i < kFramesInFlight; ++i)
    {
        _vertexBuffer[i].Destroy(_vk);
        _indexBuffer[i].Destroy(_vk);
        _uboBuffer[i].Destroy(_vk);
        _dynMeshBuffer[i].Destroy(_vk);
        _stagingBuffer[i].Destroy(_vk);
    }
}

bool EngineVulkan::CreateDepthTarget()
{
    // Pick the depth-stencil format once; pipelines bake it in, so it must
    // not change across swapchain rebuilds.
    if (_depthFormat == vk::Format::eUndefined)
    {
        for (const vk::Format candidate : {vk::Format::eD24UnormS8Uint, vk::Format::eD32SfloatS8Uint})
        {
            const vk::FormatProperties props = _vk.physicalDevice.getFormatProperties(candidate);
            if (props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment)
            {
                _depthFormat = candidate;
                break;
            }
        }
        if (_depthFormat == vk::Format::eUndefined)
        {
            LOG_ERROR(Graphics, "VK: no supported depth-stencil format");
            return false;
        }
    }

    vk::ImageCreateInfo info({}, vk::ImageType::e2D, _depthFormat,
                             vk::Extent3D(_swapchain.extent.width, _swapchain.extent.height, 1), 1, 1,
                             vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
                             vk::ImageUsageFlagBits::eDepthStencilAttachment);
    VkImageCreateInfo rawInfo = info;
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    VkImage rawImage = VK_NULL_HANDLE;
    if (vmaCreateImage(_vk.allocator, &rawInfo, &allocInfo, &rawImage, &_depthAlloc, nullptr) != VK_SUCCESS)
    {
        LOG_ERROR(Graphics, "VK: depth target allocation failed");
        return false;
    }
    _depthImage = rawImage;
    _depthView = _vk.device.createImageView(
        {{},
         _depthImage,
         vk::ImageViewType::e2D,
         _depthFormat,
         {},
         vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil, 0, 1, 0, 1)});
    return true;
}

void EngineVulkan::DestroyDepthTarget()
{
    if (_depthView)
        _vk.device.destroyImageView(_depthView);
    _depthView = nullptr;
    if (_depthImage)
        vmaDestroyImage(_vk.allocator, _depthImage, _depthAlloc);
    _depthImage = nullptr;
    _depthAlloc = nullptr;
}

// ── Pipeline cache ──────────────────────────────────────────────────────────

vk::Pipeline EngineVulkan::GetOrCreatePipeline(const PipelineKey& key)
{
    if (const auto it = _pipelines.find(key); it != _pipelines.end())
        return it->second;

    using render::BlendMode;
    using render::CullMode;
    using render::DepthMode;
    using render::FrontFaceMode;

    // Stage selection: mesh input pairs VSTransform (VSShadow for the shadow
    // family) with the 3D pixel shaders; screen input always runs VSScreen.
    vk::ShaderModule vsModule = _vsScreenModule;
    if (key.mesh)
        vsModule = (key.ps == PSSel::Shadow) ? _vsShadowModule : _vsTransformModule;
    vk::ShaderModule psModule = _psNormalModule;
    switch (key.ps)
    {
        case PSSel::Normal:
            psModule = _psNormalModule;
            break;
        case PSSel::Detail:
            psModule = _psDetailModule;
            break;
        case PSSel::Grass:
            psModule = _psGrassModule;
            break;
        case PSSel::Water:
            psModule = _psWaterModule;
            break;
        case PSSel::Flat:
            psModule = _psFlatModule;
            break;
        case PSSel::Shadow:
            psModule = _psShadowModule;
            break;
    }
    const vk::PipelineShaderStageCreateInfo stages[2] = {
        {{}, vk::ShaderStageFlagBits::eVertex, vsModule, "main"},
        {{}, vk::ShaderStageFlagBits::eFragment, psModule, "main"},
    };

    // TLVertex layout, mirroring GL33's _vaoScreen attribute setup. Packed
    // colors are 0xAARRGGBB uint32 = B,G,R,A bytes — VK_FORMAT_B8G8R8A8_UNORM.
    const vk::VertexInputBindingDescription screenBinding(0, sizeof(TLVertex), vk::VertexInputRate::eVertex);
    const vk::VertexInputAttributeDescription screenAttrs[] = {
        {0, 0, vk::Format::eR32G32B32Sfloat, offsetof(TLVertex, pos)},
        {1, 0, vk::Format::eR32Sfloat, offsetof(TLVertex, rhw)},
        {2, 0, vk::Format::eB8G8R8A8Unorm, offsetof(TLVertex, color)},
        {3, 0, vk::Format::eB8G8R8A8Unorm, offsetof(TLVertex, specular)},
        {4, 0, vk::Format::eR32G32Sfloat, offsetof(TLVertex, t0)},
        {5, 0, vk::Format::eR32G32Sfloat, offsetof(TLVertex, t1)},
    };
    // SVertexVulkan layout for the mesh path (GL33's SetupSVertexLayout).
    const vk::VertexInputBindingDescription meshBinding(0, sizeof(SVertexVulkan), vk::VertexInputRate::eVertex);
    const vk::VertexInputAttributeDescription meshAttrs[] = {
        {0, 0, vk::Format::eR32G32B32Sfloat, offsetof(SVertexVulkan, pos)},
        {1, 0, vk::Format::eR32G32B32Sfloat, offsetof(SVertexVulkan, norm)},
        {2, 0, vk::Format::eR32G32Sfloat, offsetof(SVertexVulkan, t0)},
    };
    const vk::PipelineVertexInputStateCreateInfo vertexInput =
        key.mesh ? vk::PipelineVertexInputStateCreateInfo({}, 1, &meshBinding, 3, meshAttrs)
                 : vk::PipelineVertexInputStateCreateInfo({}, 1, &screenBinding, 6, screenAttrs);

    const vk::PipelineInputAssemblyStateCreateInfo inputAssembly({}, vk::PrimitiveTopology::eTriangleList);
    const vk::PipelineViewportStateCreateInfo viewportState({}, 1, nullptr, 1, nullptr);

    vk::PipelineRasterizationStateCreateInfo raster;
    raster.lineWidth = 1.0f;
    switch (static_cast<CullMode>(key.cull))
    {
        case CullMode::Back:
            raster.cullMode = vk::CullModeFlagBits::eBack;
            break;
        case CullMode::Front:
            raster.cullMode = vk::CullModeFlagBits::eFront;
            break;
        case CullMode::None:
            raster.cullMode = vk::CullModeFlagBits::eNone;
            break;
    }
    // Direct mapping, no inversion: the negative-height viewport reproduces
    // GL's on-screen image exactly, and both APIs define CW/CCW by the
    // *visual* orientation (GL in y-up window coords, Vulkan's area formula
    // has the sign flip for y-down built in) — so what GL called front-CW is
    // front-CW here too.
    raster.frontFace = (static_cast<FrontFaceMode>(key.frontFace) == FrontFaceMode::CW)
                           ? vk::FrontFace::eClockwise
                           : vk::FrontFace::eCounterClockwise;

    const vk::PipelineMultisampleStateCreateInfo multisample;

    vk::PipelineDepthStencilStateCreateInfo depthStencil;
    depthStencil.depthCompareOp = vk::CompareOp::eLessOrEqual;
    // Per-poly shadow exclusion, mirroring GL33's GLDepthStencilState:
    // non-shadow draws stamp stencil 0 (ALWAYS + REPLACE ref=0), shadow
    // polys draw with EQUAL 0 + INCR_SAT so a pixel covered by several
    // overlapping shadow polys (knees, elbows) darkens exactly once.
    depthStencil.stencilTestEnable = true;
    vk::StencilOpState stencilOp;
    stencilOp.failOp = vk::StencilOp::eKeep;
    stencilOp.depthFailOp = vk::StencilOp::eKeep;
    stencilOp.compareMask = 0xFF;
    stencilOp.writeMask = 0xFF;
    stencilOp.reference = 0;
    switch (static_cast<DepthMode>(key.depth))
    {
        case DepthMode::Normal:
            depthStencil.depthTestEnable = true;
            depthStencil.depthWriteEnable = true;
            break;
        case DepthMode::ReadOnly:
            depthStencil.depthTestEnable = true;
            depthStencil.depthWriteEnable = false;
            break;
        case DepthMode::Disabled:
            break;
        case DepthMode::Shadow:
            depthStencil.depthTestEnable = true;
            depthStencil.depthWriteEnable = false;
            break;
    }
    if (static_cast<DepthMode>(key.depth) == DepthMode::Shadow)
    {
        stencilOp.compareOp = vk::CompareOp::eEqual;
        stencilOp.passOp = vk::StencilOp::eIncrementAndClamp;
    }
    else
    {
        stencilOp.compareOp = vk::CompareOp::eAlways;
        stencilOp.passOp = vk::StencilOp::eReplace;
    }
    depthStencil.front = stencilOp;
    depthStencil.back = stencilOp;

    vk::PipelineColorBlendAttachmentState blendAttachment;
    blendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                                     vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    blendAttachment.colorBlendOp = vk::BlendOp::eAdd;
    blendAttachment.alphaBlendOp = vk::BlendOp::eAdd;
    switch (static_cast<BlendMode>(key.blend))
    {
        case BlendMode::Opaque:
            blendAttachment.blendEnable = false;
            break;
        case BlendMode::AlphaBlend:
            blendAttachment.blendEnable = true;
            blendAttachment.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
            blendAttachment.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
            blendAttachment.srcAlphaBlendFactor = vk::BlendFactor::eSrcAlpha;
            blendAttachment.dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
            break;
        case BlendMode::Additive:
            blendAttachment.blendEnable = true;
            blendAttachment.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha;
            blendAttachment.dstColorBlendFactor = vk::BlendFactor::eOne;
            blendAttachment.srcAlphaBlendFactor = vk::BlendFactor::eSrcAlpha;
            blendAttachment.dstAlphaBlendFactor = vk::BlendFactor::eOne;
            break;
        case BlendMode::Shadow:
            blendAttachment.blendEnable = true;
            blendAttachment.srcColorBlendFactor = vk::BlendFactor::eZero;
            blendAttachment.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
            blendAttachment.srcAlphaBlendFactor = vk::BlendFactor::eZero;
            blendAttachment.dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
            break;
    }
    const vk::PipelineColorBlendStateCreateInfo blend({}, false, vk::LogicOp::eCopy, 1, &blendAttachment);

    const vk::DynamicState dynamics[2] = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    const vk::PipelineDynamicStateCreateInfo dynamicState({}, 2, dynamics);

    const vk::Format colorFormat = _swapchain.format;
    vk::PipelineRenderingCreateInfo rendering(0, 1, &colorFormat);
    rendering.depthAttachmentFormat = _depthFormat;
    rendering.stencilAttachmentFormat = _depthFormat;

    vk::GraphicsPipelineCreateInfo info({}, 2, stages, &vertexInput, &inputAssembly, nullptr, &viewportState, &raster,
                                        &multisample, &depthStencil, &blend, &dynamicState, _pipelineLayout);
    info.pNext = &rendering;

    vk::Pipeline pipeline;
    try
    {
        pipeline = _vk.device.createGraphicsPipeline(nullptr, info).value;
    }
    catch (const std::exception& e)
    {
        LOG_ERROR(Graphics, "VK: pipeline creation failed (ps={} blend={} depth={}): {}", (int)key.ps, key.blend,
                  key.depth, e.what());
        return nullptr;
    }
    _pipelines.emplace(key, pipeline);
    LOG_DEBUG(Graphics, "VK: pipeline created (ps={} blend={} depth={} cull={} ff={}) — cache size {}", (int)key.ps,
              key.blend, key.depth, key.cull, key.frontFace, _pipelines.size());
    return pipeline;
}

// ── Per-flush state application (GL33's ApplyPassState analogue) ───────────

void EngineVulkan::ApplyPassState(Texture* tex, int level, const render::LegacySpec& spec, PassId passId,
                                  PipelineVertexInput vertexInput)
{
    (void)level;
    const bool meshInput = vertexInput == PipelineVertexInput::Mesh ||
                           (vertexInput == PipelineVertexInput::ActivePass && IsIn3DPass());

    // Resolve the flush's texture view now (the surface pointer may move
    // between queue-fill and flush); null keeps the white 1x1 fallback.
    _currentTexView = nullptr;
    if (tex)
        _currentTexView = static_cast<TextureVulkan*>(tex)->GetSampledView();

    render::BuildContext ctx;
    ctx.isIn3DPass = meshInput;
    ctx.isMultitexturing = IsMultitexturing();
    ctx.shadowAlphaRef = static_cast<std::uint8_t>((_shadowFactor * 7) >> 4);
    ctx.passKindHint = GetPassKindHint();
    const render::RenderPassDescriptor d = render::BuildRenderPassDescriptor(spec, ctx);
    (void)passId;

    _currentSamplerIdx = (d.sampler.filter == render::SamplerFilter::Point ? 4 : 0) + (d.sampler.clampU ? 2 : 0) +
                         (d.sampler.clampV ? 1 : 0);

    SetShaderFogEnabled(d.fog == render::FogMode::Enabled);
    const bool alphaTest = (d.alpha == render::AlphaMode::Test || d.alpha == render::AlphaMode::TestAndBlend);
    SetAlphaTest(alphaTest, d.alphaRef);

    // TexGen constants (GL33's DoEnableDetailTexGen mirror).
    UploadVSTexGenConstants(d.texGen);

    PipelineKey key;
    // Shader family -> pixel shader; the vertex stage is derived from the
    // (mesh, ps) pair inside GetOrCreatePipeline. On the screen path the 3D
    // families can't occur; Shadow keeps its cutout PS on both paths.
    key.mesh = meshInput ? 1 : 0;
    switch (d.shader)
    {
        case render::ShaderFamily::Shadow:
            key.ps = PSSel::Shadow;
            break;
        case render::ShaderFamily::Water:
            key.ps = PSSel::Water;
            break;
        case render::ShaderFamily::Detail:
            key.ps = PSSel::Detail;
            break;
        case render::ShaderFamily::Grass:
            key.ps = PSSel::Grass;
            break;
        case render::ShaderFamily::Flat:
            key.ps = PSSel::Flat;
            break;
        case render::ShaderFamily::Normal:
        default:
            key.ps = PSSel::Normal;
            break;
    }
    key.blend = static_cast<uint8_t>(d.blend);
    key.depth = static_cast<uint8_t>(d.depth);
    key.cull = static_cast<uint8_t>(d.cull);
    key.frontFace = static_cast<uint8_t>(d.frontFace);

    // TEXTURE1 slot (GL33's SetMultiTexturing mirror): resolve the bank's
    // detail/grass/specular texture from the spec's backend bits; the
    // grass PS constants ride along like GL33's SelectPixelShader hook.
    _currentTex1View = nullptr;
    if (key.ps == PSSel::Grass)
        DoSetGrassParamsPS();
    else if (key.ps == PSSel::Water && GScene)
    {
        LightSun* sun = GScene->MainLight();
        const float lightDir[4] = {sun->SunDirection().X(), sun->SunDirection().Y(), sun->SunDirection().Z(), 0};
        UploadPSConstant(4 /*SlotLightDir*/, lightDir);
    }
    if (tex)
    {
        constexpr render::Backend mtMask =
            render::Backend::DetailTexture | render::Backend::SpecularTexture | render::Backend::GrassTexture;
        if ((spec.backend & mtMask) != render::Backend::None)
        {
            TextBankVulkan* bank = static_cast<TextBankVulkan*>(_bank);
            TextureVulkan* tex1 = nullptr;
            if (render::Has(spec.backend, render::Backend::GrassTexture))
                tex1 = bank->GetGrassTexture();
            else if (render::Has(spec.backend, render::Backend::DetailTexture))
                tex1 = bank->GetDetailTexture();
            else
                tex1 = bank->GetSpecularTexture();
            if (tex1)
            {
                bank->UseMipmap(tex1, 0, 0);
                _currentTex1View = tex1->GetSampledView();
            }
        }
    }

    if (!_pipelineBound || !(key == _lastBoundKey))
    {
        if (const vk::Pipeline pipeline = GetOrCreatePipeline(key))
        {
            _frames[_frameIndex].cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
            _lastBoundKey = key;
            _pipelineBound = true;
        }
    }
}

bool EngineVulkan::WriteConstantsAndBindDescriptors(vk::CommandBuffer cmd)
{
    // Fast path: constants and textures unchanged since the last write —
    // rebind the cached set (the world matrix is a push constant, so mesh
    // draws with the same material land here).
    const vk::ImageView tex0 = _currentTexView ? _currentTexView : _whiteView;
    const vk::ImageView tex1 = _currentTex1View ? _currentTex1View : _whiteView;
    if (!_constDirty && _cachedSet && _cachedTex0 == tex0 && _cachedTex1 == tex1 &&
        _cachedSampler == _currentSamplerIdx)
    {
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, _pipelineLayout, 0, _cachedSet,
                               {_cachedDynOffsets[0], _cachedDynOffsets[1], _instOffset});
        return true;
    }

    VulkanBuffer& ring = _uboBuffer[_frameIndex];
    vk::DeviceSize vsOffset = 0, psOffset = 0;
    void* vsDst = ring.Allocate(sizeof(_vsConst), _uboAlign, vsOffset);
    void* psDst = ring.Allocate(sizeof(_psConst), _uboAlign, psOffset);
    if (!vsDst || !psDst)
    {
        if (!_soupOverflowLogged)
        {
            LOG_ERROR(Graphics, "VK: UBO ring exhausted — draws dropped for the rest of the frame");
            _soupOverflowLogged = true;
        }
        return false;
    }
    memcpy(vsDst, _vsConst, sizeof(_vsConst));
    memcpy(psDst, _psConst, sizeof(_psConst));

    // Fresh descriptor set per flush from the frame's pool (reset in InitDraw).
    vk::DescriptorSet set;
    try
    {
        const vk::DescriptorSetAllocateInfo allocInfo(_frameDescPool[_frameIndex], 1, &_setLayout);
        set = _vk.device.allocateDescriptorSets(allocInfo).front();
    }
    catch (const std::exception&)
    {
        if (!_soupOverflowLogged)
        {
            LOG_ERROR(Graphics, "VK: frame descriptor pool exhausted — draws dropped for the rest of the frame");
            _soupOverflowLogged = true;
        }
        return false;
    }

    const vk::DescriptorBufferInfo vsInfo(ring.buffer, 0, kVSConstFloats * sizeof(float));
    const vk::DescriptorBufferInfo psInfo(ring.buffer, 0, kPSConstFloats * sizeof(float));
    const vk::DescriptorBufferInfo instInfo(ring.buffer, 0, kWorldInstancesBytes);
    const vk::DescriptorImageInfo texInfo(_samplers[_currentSamplerIdx], tex0,
                                          vk::ImageLayout::eShaderReadOnlyOptimal);
    // TEXTURE1 keeps the default linear-wrap sampler like GL33 (only slot 0
    // gets the per-draw sampler state).
    const vk::DescriptorImageInfo tex1Info(_samplers[0], tex1, vk::ImageLayout::eShaderReadOnlyOptimal);
    // Cascade shadow array (binding 5). The real array only after a depth
    // pass has put it into shader-read layout; the cleared 1x1 fallback
    // otherwise. shadowCtl.x gates the actual sampling.
    const vk::ImageView shadowView = (_shadowMapActive && _shadowArrayView) ? _shadowArrayView : _shadowFallbackView;
    const vk::DescriptorImageInfo shadowInfo(_shadowSampler, shadowView, vk::ImageLayout::eShaderReadOnlyOptimal);
    const vk::WriteDescriptorSet writes[] = {
        {set, 0, 0, 1, vk::DescriptorType::eUniformBufferDynamic, nullptr, &vsInfo},
        {set, 1, 0, 1, vk::DescriptorType::eUniformBufferDynamic, nullptr, &psInfo},
        {set, 2, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texInfo},
        {set, 3, 0, 1, vk::DescriptorType::eCombinedImageSampler, &tex1Info},
        {set, 4, 0, 1, vk::DescriptorType::eUniformBufferDynamic, nullptr, &instInfo},
        {set, 5, 0, 1, vk::DescriptorType::eCombinedImageSampler, &shadowInfo},
    };
    _vk.device.updateDescriptorSets(6, writes, 0, nullptr);

    const uint32_t dynamicOffsets[3] = {(uint32_t)vsOffset, (uint32_t)psOffset, _instOffset};
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, _pipelineLayout, 0, set, dynamicOffsets);

    _cachedSet = set;
    _cachedDynOffsets[0] = dynamicOffsets[0];
    _cachedDynOffsets[1] = dynamicOffsets[1];
    _cachedTex0 = tex0;
    _cachedTex1 = tex1;
    _cachedSampler = _currentSamplerIdx;
    _constDirty = false;
    return true;
}

} // namespace Poseidon
