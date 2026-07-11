// Frame cycle: 
// InitDraw = wait fence -> acquire -> begin rendering
// FinishDraw = end rendering 
// NextFrame = submit + present
#include "EngineVulkan.hpp"

#include <Poseidon/Foundation/Logging/Logging.hpp>

namespace
{

void TransitionImage(vk::CommandBuffer cmd, vk::Image image, vk::ImageLayout from, vk::ImageLayout to,
                     vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess, vk::PipelineStageFlags2 dstStage,
                     vk::AccessFlags2 dstAccess)
{
    vk::ImageMemoryBarrier2 barrier(srcStage, srcAccess, dstStage, dstAccess, from, to, vk::QueueFamilyIgnored,
                                    vk::QueueFamilyIgnored, image,
                                    vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
    cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, barrier));
}

vk::ClearColorValue ToClearColor(Poseidon::PackedColor color)
{
    return vk::ClearColorValue(((color >> 16) & 0xFF) / 255.0f, ((color >> 8) & 0xFF) / 255.0f,
                               (color & 0xFF) / 255.0f, ((color >> 24) & 0xFF) / 255.0f);
}

} // namespace

namespace Poseidon
{

bool EngineVulkan::CreateFrameResources()
{
    try
    {
        for (auto& f : _frames)
        {
            f.pool = _vk.device.createCommandPool(
                vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, _vk.graphicsQueueFamily));
            const auto cmds = _vk.device.allocateCommandBuffers(
                vk::CommandBufferAllocateInfo(f.pool, vk::CommandBufferLevel::ePrimary, 2));
            f.cmd = cmds[0];
            f.uploadCmd = cmds[1];
            // Signaled, or the first InitDraw would wait forever
            f.inFlight = _vk.device.createFence(vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled));
            f.imageAvailable = _vk.device.createSemaphore({});
        }
        return true;
    }
    catch (const std::exception& e)
    {
        LOG_ERROR(Graphics, "VK: frame resource creation failed: {}", e.what());
        return false;
    }
}

void EngineVulkan::DestroyFrameResources()
{
    for (auto& f : _frames)
    {
        if (f.pool)
            _vk.device.destroyCommandPool(f.pool);
        if (f.inFlight)
            _vk.device.destroyFence(f.inFlight);
        if (f.imageAvailable)
            _vk.device.destroySemaphore(f.imageAvailable);
        f = {};
    }
}

void EngineVulkan::RecreateSwapchain()
{
    _vk.device.waitIdle();
    if (_swapchain.Create(_vk, _sdlWindow, _swapInterval))
    {
        _swapchainDirty = false;
        _w = (int)_swapchain.extent.width;
        _h = (int)_swapchain.extent.height;
        DestroyDepthTarget();
        if (!CreateDepthTarget())
            LOG_ERROR(Graphics, "VK: depth target recreation failed");
    }
    // While minimized _swapchainDirty stays set; retried next frame
}

bool EngineVulkan::InitDrawDone()
{
    return _frameOpen;
}

void EngineVulkan::InitDraw(bool clear, PackedColor color)
{
    if (_frameOpen)
    {
        LOG_DEBUG(Graphics, "InitDraw done twice");
        return;
    }
    if (!IsUsable())
        return;
    if (clear)
        _clearColor = color;

    FrameResources& f = _frames[_frameIndex];
    (void)_vk.device.waitForFences(f.inFlight, true, UINT64_MAX);

    if (_swapchainDirty)
        RecreateSwapchain();
    if (!_swapchain.IsValid())
        return;

    try
    {
        const vk::Result r =
            _vk.device.acquireNextImageKHR(_swapchain.swapchain, UINT64_MAX, f.imageAvailable, nullptr, &_imageIndex);
        if (r == vk::Result::eSuboptimalKHR)
            _swapchainDirty = true; // image still usable this frame
    }
    catch (const vk::OutOfDateKHRError&)
    {
        _swapchainDirty = true; // skip this frame
        return;
    }

    // Reset only after acquire succeeded: every early-out above must leave the fence signaled or the next InitDraw deadlocks
    _vk.device.resetFences(f.inFlight);

    f.cmd.reset();
    f.cmd.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

    TransitionImage(f.cmd, _swapchain.images[_imageIndex], vk::ImageLayout::eUndefined,
                    vk::ImageLayout::eColorAttachmentOptimal, vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                    vk::AccessFlags2{}, vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                    vk::AccessFlagBits2::eColorAttachmentWrite);

    // Depth is transient per frame: contents discarded via eUndefined, cleared
    // by the attachment loadOp.
    {
        vk::ImageMemoryBarrier2 depthBarrier(
            vk::PipelineStageFlagBits2::eLateFragmentTests, vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
            vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::QueueFamilyIgnored,
            vk::QueueFamilyIgnored, _depthImage,
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil, 0, 1, 0,
                                      1));
        f.cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, depthBarrier));
    }

    vk::RenderingAttachmentInfo colorAttachment(_swapchain.views[_imageIndex],
                                                vk::ImageLayout::eColorAttachmentOptimal);
    colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
    colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
    colorAttachment.clearValue = vk::ClearValue(ToClearColor(_clearColor));

    vk::RenderingAttachmentInfo depthAttachment(_depthView, vk::ImageLayout::eDepthStencilAttachmentOptimal);
    depthAttachment.loadOp = vk::AttachmentLoadOp::eClear;
    depthAttachment.storeOp = vk::AttachmentStoreOp::eStore;
    depthAttachment.clearValue = vk::ClearValue(vk::ClearDepthStencilValue(1.0f, 0));

    vk::RenderingInfo renderingInfo({}, vk::Rect2D({0, 0}, _swapchain.extent), 1, 0, colorAttachment);
    renderingInfo.pDepthAttachment = &depthAttachment;
    renderingInfo.pStencilAttachment = &depthAttachment;
    f.cmd.beginRendering(renderingInfo);

    // GL-convention rasterization: negative-height viewport (doc 4.6), so the
    // ported shaders and winding work without changes.
    f.cmd.setViewport(0, vk::Viewport(0.0f, (float)_swapchain.extent.height, (float)_swapchain.extent.width,
                                      -(float)_swapchain.extent.height, 0.0f, 1.0f));
    f.cmd.setScissor(0, vk::Rect2D({0, 0}, _swapchain.extent));

    // Reopen the per-frame allocators: the fence wait above guarantees the
    // GPU is done reading this slot's buffers.
    _vertexBuffer[_frameIndex].Reset();
    _indexBuffer[_frameIndex].Reset();
    _uboBuffer[_frameIndex].Reset();
    _dynMeshBuffer[_frameIndex].Reset();
    _stagingBuffer[_frameIndex].Reset();
    _vk.device.resetDescriptorPool(_frameDescPool[_frameIndex]);
    FlushDeferredDestroys(_frameIndex);
    _vertexWindowBase = 0;
    _indexWindowBase = 0;
    _queueNo._vertexBufferUsed = 0;
    _queueNo._indexBufferUsed = 0;
    _soupOverflowLogged = false;
    _pipelineBound = false;
    _activePassId = PassId::ScreenSpace;
    if (_bank)
        _bank->StartFrame(); // rotate the texture frame-LRU lists (mirrors GL33)
    // Base per-frame state: recomputes _accomodateEye = HWhite * userBrightness
    // (default 1.6) and the night-vision green filter. Skipping this left the
    // whole scene exactly 0.625x (1/1.6) darker than GL33.
    Engine::InitDraw(clear, color);
    InvalidateMaterialCache(); // per-frame lighting inputs may have changed
    _cachedSet = nullptr;      // the pool reset above freed it
    _constDirty = true;
    _instCount = 0;
    _instImpure = false;
    _instOffset = 0; // last frame's slice retired with its ring
    UploadVSScreenConstants();

    _frameOpen = true;
}

void EngineVulkan::Clear(bool clearZ, bool clear, PackedColor color)
{
    if (clear)
        _clearColor = color;
    if (!_frameOpen || (!clear && !clearZ))
        return;

    vk::ClearAttachment attachments[2];
    uint32_t count = 0;
    if (clear)
        attachments[count++] = vk::ClearAttachment(vk::ImageAspectFlagBits::eColor, 0, vk::ClearValue(ToClearColor(color)));
    if (clearZ)
        attachments[count++] =
            vk::ClearAttachment(vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil, 0,
                                vk::ClearValue(vk::ClearDepthStencilValue(1.0f, 0)));
    vk::ClearRect rect(vk::Rect2D({0, 0}, _swapchain.extent), 0, 1);
    _frames[_frameIndex].cmd.clearAttachments(count, attachments, 1, &rect);
}

void EngineVulkan::FinishDraw()
{
    if (!_frameOpen)
        return;

    // Base frame bookkeeping (frame counter, durations) + debug texts,
    // then commit whatever the producer left queued (mirrors GL33).
    Engine::FinishDraw();
    Engine::DrawFinishTexts();
    FlushAndFreeAllQueues(_queueNo);

    FrameResources& f = _frames[_frameIndex];

    // Dev overlay composites on top of the finished game frame, inside the
    // still-open rendering pass (mirrors GL33's BackToFront timing).
    RenderDebugOverlay(f.cmd);

    f.cmd.endRendering();

    TransitionImage(f.cmd, _swapchain.images[_imageIndex], vk::ImageLayout::eColorAttachmentOptimal,
                    vk::ImageLayout::ePresentSrcKHR, vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                    vk::AccessFlagBits2::eColorAttachmentWrite, vk::PipelineStageFlagBits2::eNone, vk::AccessFlags2{});
    f.cmd.end();

    _frameOpen = false;
    _frameRecorded = true;

    if (_bank)
        _bank->FinishFrame(); // texture LRU bookkeeping (mirrors GL33)
}

void EngineVulkan::NextFrame()
{
    FrameResources& f = _frames[_frameIndex];

    if (!_frameRecorded)
    {
        // Frame was skipped after uploads were already recorded (minimize /
        // out-of-date acquire): submit the uploads alone so the CB doesn't
        // stay in recording state; slot fence tracks them for the resets.
        if (_uploadOpen)
        {
            f.uploadCmd.end();
            _uploadOpen = false;
            _vk.device.resetFences(f.inFlight);
            vk::CommandBufferSubmitInfo uploadInfo(f.uploadCmd);
            _vk.graphicsQueue.submit2(vk::SubmitInfo2({}, {}, uploadInfo), f.inFlight);
            _frameIndex = (_frameIndex + 1) % kFramesInFlight;
        }
        return;
    }

    // Texture uploads recorded this frame execute before the draws that
    // sample them; the image barriers inside uploadCmd order the accesses.
    vk::CommandBufferSubmitInfo cmdInfos[2];
    uint32_t cmdCount = 0;
    if (_uploadOpen)
    {
        f.uploadCmd.end();
        _uploadOpen = false;
        cmdInfos[cmdCount++] = vk::CommandBufferSubmitInfo(f.uploadCmd);
    }
    cmdInfos[cmdCount++] = vk::CommandBufferSubmitInfo(f.cmd);

    vk::SemaphoreSubmitInfo waitAcquire(f.imageAvailable, 0, vk::PipelineStageFlagBits2::eColorAttachmentOutput);
    vk::SemaphoreSubmitInfo signalRendered(_swapchain.renderFinished[_imageIndex], 0,
                                           vk::PipelineStageFlagBits2::eAllCommands);
    vk::SubmitInfo2 submit({}, 1, &waitAcquire, cmdCount, cmdInfos, 1, &signalRendered);
    _vk.graphicsQueue.submit2(submit, f.inFlight);

    try
    {
        const vk::Result r = _vk.graphicsQueue.presentKHR(
            vk::PresentInfoKHR(_swapchain.renderFinished[_imageIndex], _swapchain.swapchain, _imageIndex));
        if (r == vk::Result::eSuboptimalKHR)
            _swapchainDirty = true;
    }
    catch (const vk::OutOfDateKHRError&)
    {
        _swapchainDirty = true;
    }

    _frameRecorded = false;
    _frameIndex = (_frameIndex + 1) % kFramesInFlight;
}

void EngineVulkan::OnWindowResized(int /*w*/, int /*h*/)
{
    _swapchainDirty = true; // sizes are re-derived from the surface on rebuild
}

} // namespace Poseidon
