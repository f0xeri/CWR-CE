// Cascade shadow-map depth pass — Vulkan port of EngineGL33_ShadowDepth.cpp.
// Scene calls RenderShadowDepthScene mid-frame (DrawObjectsAndShadowsPass2)
// with the collected casters and per-cascade light view-projections; we
// suspend the frame's main dynamic-rendering pass, render the casters into
// each layer of a depth array from the light, and resume. The lit shaders
// sample the array at binding 5 (constants fed by UpdateShadowMapLitState at
// the next 3D-pass begin, mirroring GL33's one-frame constant latency).
//
// Y convention: unlike the main pass (negative-height viewport to reproduce
// GL rasterization), the depth pass uses a standard Vulkan viewport. Writing
// AND sampling both map clip.xy -> uv with the same formula (suv = sc*0.5+0.5
// in the lit shaders), so the pair is self-consistent without any flip. The
// flip does mirror screen-space winding, so "cull the front faces" (GL33's
// anti-acne trick) becomes frontFace=CCW + cullMode=Front here.
#include "EngineVulkan.hpp"
#include "TextureVulkan.hpp"
#include "Utils/VulkanShaderCompiler.hpp"

#include <Poseidon/Foundation/Logging/Logging.hpp>

#include <cstring>
#include <vector>

namespace
{

// Depth-only caster programs; the light view-projection rides as a push
// constant (one mat4), like GL33's uLightVP uniform.
const char* const kShadowSolidVS = R"(#version 450
layout(push_constant) uniform PushVP { mat4 lightVP; } pc;
layout(location = 0) in vec3 pos;
void main() { gl_Position = pc.lightVP * vec4(pos, 1.0); }
)";

const char* const kShadowSolidPS = R"(#version 450
void main() {}
)";

const char* const kShadowAlphaVS = R"(#version 450
layout(push_constant) uniform PushVP { mat4 lightVP; } pc;
layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 uv;
layout(location = 0) out vec2 vUV;
void main() { vUV = uv; gl_Position = pc.lightVP * vec4(pos, 1.0); }
)";

const char* const kShadowAlphaPS = R"(#version 450
layout(set = 0, binding = 0) uniform sampler2D uTex;
layout(location = 0) in vec2 vUV;
void main() { if (texture(uTex, vUV).a < 0.5) discard; }
)";

} // namespace

namespace Poseidon
{

// Pick a sampleable depth format once (shared by the fallback and the real
// cascade array; pipelines bake it in).
static vk::Format PickShadowDepthFormat(VulkanContext& vk)
{
    for (const vk::Format candidate : {vk::Format::eD32Sfloat, vk::Format::eD16Unorm})
    {
        const vk::FormatProperties props = vk.physicalDevice.getFormatProperties(candidate);
        const vk::FormatFeatureFlags need =
            vk::FormatFeatureFlagBits::eDepthStencilAttachment | vk::FormatFeatureFlagBits::eSampledImage;
        if ((props.optimalTilingFeatures & need) == need)
            return candidate;
    }
    return vk::Format::eUndefined;
}

bool EngineVulkan::CreateShadowFallbackTexture()
{
    _shadowFormat = PickShadowDepthFormat(_vk);
    if (_shadowFormat == vk::Format::eUndefined)
    {
        LOG_ERROR(Graphics, "VK: no sampleable depth format for shadow maps");
        return false;
    }

    // 1x1 single-layer depth array cleared to 1.0 ("fully lit"): binding 5 is
    // statically used by the lit shaders, so a valid array view must exist
    // before (or without) any depth pass.
    vk::ImageCreateInfo info({}, vk::ImageType::e2D, _shadowFormat, vk::Extent3D(1, 1, 1), 1, 1,
                             vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
                             vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst);
    VkImageCreateInfo rawInfo = info;
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    VkImage rawImage = VK_NULL_HANDLE;
    if (vmaCreateImage(_vk.allocator, &rawInfo, &allocInfo, &rawImage, &_shadowFallbackAlloc, nullptr) != VK_SUCCESS)
    {
        LOG_ERROR(Graphics, "VK: shadow fallback allocation failed");
        return false;
    }
    _shadowFallbackImage = rawImage;
    _shadowFallbackView = _vk.device.createImageView(
        {{}, _shadowFallbackImage, vk::ImageViewType::e2DArray, _shadowFormat, {},
         vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1)});

    // One-shot clear + transition to shader-read (same pattern as the white
    // texture upload in InitPipelineResources).
    vk::CommandPool pool = _vk.device.createCommandPool(
        vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eTransient, _vk.graphicsQueueFamily));
    vk::CommandBuffer cmd = _vk.device.allocateCommandBuffers({pool, vk::CommandBufferLevel::ePrimary, 1}).front();
    cmd.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

    const vk::ImageSubresourceRange range(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1);
    vk::ImageMemoryBarrier2 toDst(vk::PipelineStageFlagBits2::eNone, {}, vk::PipelineStageFlagBits2::eClear,
                                  vk::AccessFlagBits2::eTransferWrite, vk::ImageLayout::eUndefined,
                                  vk::ImageLayout::eTransferDstOptimal, vk::QueueFamilyIgnored, vk::QueueFamilyIgnored,
                                  _shadowFallbackImage, range);
    cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, toDst));
    cmd.clearDepthStencilImage(_shadowFallbackImage, vk::ImageLayout::eTransferDstOptimal,
                               vk::ClearDepthStencilValue(1.0f, 0), range);
    vk::ImageMemoryBarrier2 toRead(vk::PipelineStageFlagBits2::eClear, vk::AccessFlagBits2::eTransferWrite,
                                   vk::PipelineStageFlagBits2::eFragmentShader,
                                   vk::AccessFlagBits2::eShaderSampledRead, vk::ImageLayout::eTransferDstOptimal,
                                   vk::ImageLayout::eShaderReadOnlyOptimal, vk::QueueFamilyIgnored,
                                   vk::QueueFamilyIgnored, _shadowFallbackImage, range);
    cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, toRead));

    cmd.end();
    vk::CommandBufferSubmitInfo cmdInfo(cmd);
    _vk.graphicsQueue.submit2(vk::SubmitInfo2({}, {}, cmdInfo), nullptr);
    _vk.graphicsQueue.waitIdle();
    _vk.device.destroyCommandPool(pool);
    return true;
}

bool EngineVulkan::EnsureShadowPassResources()
{
    if (_shadowSolidPipeline && _shadowAlphaPipeline)
        return true;
    if (_shadowInitFailed)
        return false;

    try
    {
        auto build = [&](vk::ShaderStageFlagBits stage, const char* src, const char* name, vk::ShaderModule& out) {
            const std::vector<uint32_t> spirv = CompileGlslToSpirv(stage, src, name);
            if (spirv.empty())
                return false;
            out = _vk.device.createShaderModule({{}, spirv.size() * sizeof(uint32_t), spirv.data()});
            return true;
        };
        if (!build(vk::ShaderStageFlagBits::eVertex, kShadowSolidVS, "vsShadowDepth", _shadowSolidVSModule) ||
            !build(vk::ShaderStageFlagBits::eFragment, kShadowSolidPS, "psShadowDepth", _shadowSolidPSModule) ||
            !build(vk::ShaderStageFlagBits::eVertex, kShadowAlphaVS, "vsShadowDepthAlpha", _shadowAlphaVSModule) ||
            !build(vk::ShaderStageFlagBits::eFragment, kShadowAlphaPS, "psShadowDepthAlpha", _shadowAlphaPSModule))
        {
            _shadowInitFailed = true;
            return false;
        }

        const vk::DescriptorSetLayoutBinding texBinding(0, vk::DescriptorType::eCombinedImageSampler, 1,
                                                        vk::ShaderStageFlagBits::eFragment);
        _shadowSetLayout = _vk.device.createDescriptorSetLayout({{}, 1, &texBinding});
        const vk::PushConstantRange vpRange(vk::ShaderStageFlagBits::eVertex, 0, 64);
        _shadowPipelineLayout = _vk.device.createPipelineLayout({{}, 1, &_shadowSetLayout, 1, &vpRange});

        // Shared fixed state for both depth-only pipelines.
        const vk::PipelineInputAssemblyStateCreateInfo inputAssembly({}, vk::PrimitiveTopology::eTriangleList);
        const vk::PipelineViewportStateCreateInfo viewportState({}, 1, nullptr, 1, nullptr);
        const vk::PipelineMultisampleStateCreateInfo multisample;
        vk::PipelineDepthStencilStateCreateInfo depthStencil;
        depthStencil.depthTestEnable = true;
        depthStencil.depthWriteEnable = true;
        depthStencil.depthCompareOp = vk::CompareOp::eLessOrEqual;
        const vk::PipelineColorBlendStateCreateInfo blend({}, false, vk::LogicOp::eCopy, 0, nullptr);
        const vk::DynamicState dynamics[2] = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
        const vk::PipelineDynamicStateCreateInfo dynamicState({}, 2, dynamics);
        vk::PipelineRenderingCreateInfo rendering(0, 0, nullptr);
        rendering.depthAttachmentFormat = _shadowFormat;

        // Solid casters: store the BACK faces (GL33's anti-acne trick — a lit
        // front-facing surface is then strictly nearer the light than the
        // stored depth, so it cannot self-shadow). The positive-Y viewport
        // mirrors screen winding vs the main pass, hence frontFace=CCW here.
        {
            const vk::PipelineShaderStageCreateInfo stages[2] = {
                {{}, vk::ShaderStageFlagBits::eVertex, _shadowSolidVSModule, "main"},
                {{}, vk::ShaderStageFlagBits::eFragment, _shadowSolidPSModule, "main"},
            };
            const vk::VertexInputBindingDescription binding(0, 3 * sizeof(float), vk::VertexInputRate::eVertex);
            const vk::VertexInputAttributeDescription attr(0, 0, vk::Format::eR32G32B32Sfloat, 0);
            const vk::PipelineVertexInputStateCreateInfo vertexInput({}, 1, &binding, 1, &attr);
            vk::PipelineRasterizationStateCreateInfo raster;
            raster.lineWidth = 1.0f;
            raster.cullMode = vk::CullModeFlagBits::eFront;
            raster.frontFace = vk::FrontFace::eCounterClockwise;
            vk::GraphicsPipelineCreateInfo info({}, 2, stages, &vertexInput, &inputAssembly, nullptr, &viewportState,
                                                &raster, &multisample, &depthStencil, &blend, &dynamicState,
                                                _shadowPipelineLayout);
            info.pNext = &rendering;
            _shadowSolidPipeline = _vk.device.createGraphicsPipeline(nullptr, info).value;
        }

        // Alpha-cutout casters (foliage): two-sided, texture-alpha discard.
        {
            const vk::PipelineShaderStageCreateInfo stages[2] = {
                {{}, vk::ShaderStageFlagBits::eVertex, _shadowAlphaVSModule, "main"},
                {{}, vk::ShaderStageFlagBits::eFragment, _shadowAlphaPSModule, "main"},
            };
            const vk::VertexInputBindingDescription binding(0, 5 * sizeof(float), vk::VertexInputRate::eVertex);
            const vk::VertexInputAttributeDescription attrs[2] = {
                {0, 0, vk::Format::eR32G32B32Sfloat, 0},
                {1, 0, vk::Format::eR32G32Sfloat, 3 * sizeof(float)},
            };
            const vk::PipelineVertexInputStateCreateInfo vertexInput({}, 1, &binding, 2, attrs);
            vk::PipelineRasterizationStateCreateInfo raster;
            raster.lineWidth = 1.0f;
            raster.cullMode = vk::CullModeFlagBits::eNone;
            raster.frontFace = vk::FrontFace::eCounterClockwise;
            vk::GraphicsPipelineCreateInfo info({}, 2, stages, &vertexInput, &inputAssembly, nullptr, &viewportState,
                                                &raster, &multisample, &depthStencil, &blend, &dynamicState,
                                                _shadowPipelineLayout);
            info.pNext = &rendering;
            _shadowAlphaPipeline = _vk.device.createGraphicsPipeline(nullptr, info).value;
        }
        return true;
    }
    catch (const std::exception& e)
    {
        LOG_ERROR(Graphics, "VK: shadow depth-pass resource init failed: {}", e.what());
        _shadowInitFailed = true;
        return false;
    }
}

bool EngineVulkan::EnsureShadowMapTarget(int res)
{
    if (_shadowImage && _shadowMapRes == res)
        return true;

    // Resolution change: earlier draws this frame may reference the old array
    // view in already-written descriptor sets — retire it через the fence-gated
    // deferred-destroy path, never inline.
    if (_shadowImage)
    {
        DeferDestroyImage(_shadowImage, _shadowAlloc, _shadowArrayView);
        for (auto& v : _shadowLayerViews)
        {
            DeferDestroyImage(nullptr, nullptr, v);
            v = nullptr;
        }
        _shadowImage = nullptr;
        _shadowAlloc = nullptr;
        _shadowArrayView = nullptr;
    }

    try
    {
        vk::ImageCreateInfo info({}, vk::ImageType::e2D, _shadowFormat,
                                 vk::Extent3D((uint32_t)res, (uint32_t)res, 1), 1, kShadowCascades,
                                 vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
                                 vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled);
        VkImageCreateInfo rawInfo = info;
        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
        allocInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
        VkImage rawImage = VK_NULL_HANDLE;
        if (vmaCreateImage(_vk.allocator, &rawInfo, &allocInfo, &rawImage, &_shadowAlloc, nullptr) != VK_SUCCESS)
        {
            LOG_ERROR(Graphics, "VK: cascade depth array allocation failed ({}x{}x{})", res, res, kShadowCascades);
            return false;
        }
        _shadowImage = rawImage;
        _shadowArrayView = _vk.device.createImageView(
            {{}, _shadowImage, vk::ImageViewType::e2DArray, _shadowFormat, {},
             vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, kShadowCascades)});
        for (int i = 0; i < kShadowCascades; i++)
        {
            _shadowLayerViews[i] = _vk.device.createImageView(
                {{}, _shadowImage, vk::ImageViewType::e2D, _shadowFormat, {},
                 vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, (uint32_t)i, 1)});
        }
        _shadowMapRes = res;
        return true;
    }
    catch (const std::exception& e)
    {
        LOG_ERROR(Graphics, "VK: cascade depth array creation failed: {}", e.what());
        return false;
    }
}

void EngineVulkan::DestroyShadowResources()
{
    // Device is idle here (DestroyPipelineResources contract).
    for (vk::Pipeline* p : {&_shadowSolidPipeline, &_shadowAlphaPipeline})
    {
        if (*p)
            _vk.device.destroyPipeline(*p);
        *p = nullptr;
    }
    if (_shadowPipelineLayout)
        _vk.device.destroyPipelineLayout(_shadowPipelineLayout);
    _shadowPipelineLayout = nullptr;
    if (_shadowSetLayout)
        _vk.device.destroyDescriptorSetLayout(_shadowSetLayout);
    _shadowSetLayout = nullptr;
    for (vk::ShaderModule* m :
         {&_shadowSolidVSModule, &_shadowSolidPSModule, &_shadowAlphaVSModule, &_shadowAlphaPSModule})
    {
        if (*m)
            _vk.device.destroyShaderModule(*m);
        *m = nullptr;
    }
    for (auto& v : _shadowLayerViews)
    {
        if (v)
            _vk.device.destroyImageView(v);
        v = nullptr;
    }
    if (_shadowArrayView)
        _vk.device.destroyImageView(_shadowArrayView);
    _shadowArrayView = nullptr;
    if (_shadowImage)
        vmaDestroyImage(_vk.allocator, _shadowImage, _shadowAlloc);
    _shadowImage = nullptr;
    _shadowAlloc = nullptr;
    if (_shadowFallbackView)
        _vk.device.destroyImageView(_shadowFallbackView);
    _shadowFallbackView = nullptr;
    if (_shadowFallbackImage)
        vmaDestroyImage(_vk.allocator, _shadowFallbackImage, _shadowFallbackAlloc);
    _shadowFallbackImage = nullptr;
    _shadowFallbackAlloc = nullptr;
    if (_shadowSampler)
        _vk.device.destroySampler(_shadowSampler);
    _shadowSampler = nullptr;
    _shadowMapActive = false;
    _shadowMapRes = 0;
}

void EngineVulkan::RenderShadowDepthScene(const float* lightVPs, const float* splitViewDist, const float* camFwd3,
                                          int numCascades, int omniCount, int res, const ShadowCasterSet& casters)
{
    if (numCascades > kShadowCascades)
        numCascades = kShadowCascades;
    const bool haveSolid = casters.solidXYZ && casters.solidVertexCount >= 3;
    const bool haveAlpha =
        casters.alphaXYZUV && casters.alphaVertexCount >= 3 && casters.alphaBatches && casters.alphaBatchCount > 0;
    if (!_frameOpen || !lightVPs || numCascades < 1 || res <= 0 || (!haveSolid && !haveAlpha))
    {
        _shadowMapActive = false;
        return;
    }
    if (!EnsureShadowPassResources() || !EnsureShadowMapTarget(res))
    {
        _shadowMapActive = false;
        return;
    }

    // Everything that can fail happens before the main pass is suspended, so
    // every early-out below this block leaves the frame untouched.
    // NOT _vertexBuffer: the soup queues address that ring directly through
    // the window counters (_vertexWindowBase + _vertexBufferUsed), bypassing
    // Allocate(), so cursor allocations would overlap the queued UI vertices.
    // _dynMeshBuffer is cursor-allocated only (AllocDynamicMeshVertices).
    VulkanBuffer& ring = _dynMeshBuffer[_frameIndex];
    vk::DeviceSize solidOffset = 0, alphaOffset = 0;
    if (haveSolid)
    {
        void* dst = ring.Allocate((vk::DeviceSize)casters.solidVertexCount * 3 * sizeof(float), 4, solidOffset);
        if (!dst)
        {
            LOG_ERROR(Graphics, "VK: vertex ring exhausted — shadow depth pass skipped");
            _shadowMapActive = false;
            return;
        }
        memcpy(dst, casters.solidXYZ, (size_t)casters.solidVertexCount * 3 * sizeof(float));
    }
    if (haveAlpha)
    {
        void* dst = ring.Allocate((vk::DeviceSize)casters.alphaVertexCount * 5 * sizeof(float), 4, alphaOffset);
        if (!dst)
        {
            LOG_ERROR(Graphics, "VK: vertex ring exhausted — shadow depth pass skipped");
            _shadowMapActive = false;
            return;
        }
        memcpy(dst, casters.alphaXYZUV, (size_t)casters.alphaVertexCount * 5 * sizeof(float));
    }

    // Resolve each alpha batch's caster texture (loading its base mip if the
    // depth pass beat the lit draw to it) and bake a descriptor set per batch,
    // like GL33's per-batch texture bind.
    struct ResolvedAlphaBatch
    {
        vk::DescriptorSet set;
        int firstVertex;
        int vertexCount;
    };
    std::vector<ResolvedAlphaBatch> alphaBatches;
    if (haveAlpha)
    {
        alphaBatches.reserve((size_t)casters.alphaBatchCount);
        for (int b = 0; b < casters.alphaBatchCount; b++)
        {
            const ShadowCasterBatch& src = casters.alphaBatches[b];
            if (src.vertexCount < 3)
                continue;
            TextureVulkan* tex = static_cast<TextureVulkan*>(src.texture);
            if (_bank && tex)
                _bank->UseMipmap(tex, 0, 0);
            vk::ImageView view = tex ? tex->GetSampledView() : vk::ImageView{};
            if (!view)
                view = _whiteView;
            vk::DescriptorSet set;
            try
            {
                const vk::DescriptorSetAllocateInfo allocInfo(_frameDescPool[_frameIndex], 1, &_shadowSetLayout);
                set = _vk.device.allocateDescriptorSets(allocInfo).front();
            }
            catch (const std::exception&)
            {
                LOG_ERROR(Graphics, "VK: descriptor pool exhausted — alpha shadow casters dropped this frame");
                break;
            }
            const vk::DescriptorImageInfo texInfo(_samplers[0], view, vk::ImageLayout::eShaderReadOnlyOptimal);
            const vk::WriteDescriptorSet write(set, 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &texInfo);
            _vk.device.updateDescriptorSets(1, &write, 0, nullptr);
            alphaBatches.push_back({set, src.firstVertex, src.vertexCount});
        }
    }

    vk::CommandBuffer cmd = _frames[_frameIndex].cmd;

    // ── Suspend the main pass and render the cascades ──────────────────────
    cmd.endRendering();

    // Prior reads (this queue, any command buffer) → attachment writes. The
    // whole array is re-cleared, so the old content is discardable (Undefined).
    const vk::ImageSubresourceRange allLayers(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, kShadowCascades);
    {
        vk::ImageMemoryBarrier2 toWrite(
            vk::PipelineStageFlagBits2::eFragmentShader | vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                vk::PipelineStageFlagBits2::eLateFragmentTests,
            vk::AccessFlagBits2::eShaderSampledRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
            vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::QueueFamilyIgnored,
            vk::QueueFamilyIgnored, _shadowImage, allLayers);
        cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, toWrite));
    }

    for (int i = 0; i < numCascades; i++)
    {
        vk::RenderingAttachmentInfo depthAttachment(_shadowLayerViews[i],
                                                    vk::ImageLayout::eDepthStencilAttachmentOptimal);
        depthAttachment.loadOp = vk::AttachmentLoadOp::eClear;
        depthAttachment.storeOp = vk::AttachmentStoreOp::eStore;
        depthAttachment.clearValue = vk::ClearValue(vk::ClearDepthStencilValue(1.0f, 0));
        vk::RenderingInfo renderingInfo;
        renderingInfo.renderArea = vk::Rect2D({0, 0}, {(uint32_t)res, (uint32_t)res});
        renderingInfo.layerCount = 1;
        renderingInfo.pDepthAttachment = &depthAttachment;
        cmd.beginRendering(renderingInfo);
        // Standard (positive) viewport — see the Y-convention note on top.
        cmd.setViewport(0, vk::Viewport(0.0f, 0.0f, (float)res, (float)res, 0.0f, 1.0f));
        cmd.setScissor(0, vk::Rect2D({0, 0}, {(uint32_t)res, (uint32_t)res}));

        if (haveSolid)
        {
            cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, _shadowSolidPipeline);
            cmd.pushConstants(_shadowPipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, 64, lightVPs + i * 16);
            cmd.bindVertexBuffers(0, ring.buffer, solidOffset);
            cmd.draw((uint32_t)casters.solidVertexCount, 1, 0, 0);
        }
        if (!alphaBatches.empty())
        {
            cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, _shadowAlphaPipeline);
            cmd.pushConstants(_shadowPipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, 64, lightVPs + i * 16);
            cmd.bindVertexBuffers(0, ring.buffer, alphaOffset);
            for (const ResolvedAlphaBatch& b : alphaBatches)
            {
                cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, _shadowPipelineLayout, 0, b.set, {});
                cmd.draw((uint32_t)b.vertexCount, 1, (uint32_t)b.firstVertex, 0);
            }
        }
        cmd.endRendering();
    }

    // Attachment writes → sampled reads in the lit shaders.
    {
        vk::ImageMemoryBarrier2 toRead(
            vk::PipelineStageFlagBits2::eLateFragmentTests, vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead,
            vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::QueueFamilyIgnored, vk::QueueFamilyIgnored, _shadowImage, allLayers);
        cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, toRead));
    }

    // ── Resume the main pass (attachments preserved via load) ──────────────
    {
        vk::RenderingAttachmentInfo colorAttachment(_swapchain.views[_imageIndex],
                                                    vk::ImageLayout::eColorAttachmentOptimal);
        colorAttachment.loadOp = vk::AttachmentLoadOp::eLoad;
        colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
        vk::RenderingAttachmentInfo depthAttachment(_depthView, vk::ImageLayout::eDepthStencilAttachmentOptimal);
        depthAttachment.loadOp = vk::AttachmentLoadOp::eLoad;
        depthAttachment.storeOp = vk::AttachmentStoreOp::eStore;
        vk::RenderingInfo renderingInfo({}, vk::Rect2D({0, 0}, _swapchain.extent), 1, 0, colorAttachment);
        renderingInfo.pDepthAttachment = &depthAttachment;
        renderingInfo.pStencilAttachment = &depthAttachment;
        cmd.beginRendering(renderingInfo);
        // Restore the frame's GL-convention negative viewport (see InitDraw).
        cmd.setViewport(0, vk::Viewport(0.0f, (float)_swapchain.extent.height, (float)_swapchain.extent.width,
                                        -(float)_swapchain.extent.height, 0.0f, 1.0f));
        cmd.setScissor(0, vk::Rect2D({0, 0}, _swapchain.extent));
    }
    // The depth pass bound its own pipelines/descriptors; force the next lit
    // draw to re-apply its state (mirrors GL33's InvalidatePipelineCache).
    _pipelineBound = false;

    // Keep the array + splits + forward + omniCount for the lit pass.
    _shadowCascades = numCascades;
    _shadowOmniCount = (omniCount < 0) ? 0 : (omniCount > numCascades ? numCascades : omniCount);
    memcpy(_shadowMapVP, lightVPs, sizeof(float) * 16 * (size_t)numCascades);
    for (int i = 0; i < numCascades; i++)
        _shadowSplits[i] = splitViewDist[i];
    _shadowCamFwd[0] = camFwd3[0];
    _shadowCamFwd[1] = camFwd3[1];
    _shadowCamFwd[2] = camFwd3[2];
    _shadowMapActive = true;

    // Refresh the lit-shader constants NOW, not only at the next 3D-pass
    // begin: draws recorded after this point sample the array content this
    // pass just wrote, so they must pair it with this pass's cascade VPs.
    // (BeginPass uploaded last frame's VPs; the cascades are texel-snapped and
    // jump every frame under camera motion, so the stale pairing reads as
    // shadow-edge shimmer. Draws recorded before this point executed against
    // the previous content + previous VPs — also a consistent pair.)
    UpdateShadowMapLitState();
}

} // namespace Poseidon
