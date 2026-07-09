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
            f.cmd = _vk.device
                        .allocateCommandBuffers(vk::CommandBufferAllocateInfo(f.pool, vk::CommandBufferLevel::ePrimary, 1))
                        .front();
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
    if (_swapchain.Create(_vk, _sdlWindow))
    {
        _swapchainDirty = false;
        _w = (int)_swapchain.extent.width;
        _h = (int)_swapchain.extent.height;
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

    vk::RenderingAttachmentInfo colorAttachment(_swapchain.views[_imageIndex],
                                                vk::ImageLayout::eColorAttachmentOptimal);
    colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
    colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
    colorAttachment.clearValue = vk::ClearValue(ToClearColor(_clearColor));
    f.cmd.beginRendering(vk::RenderingInfo({}, vk::Rect2D({0, 0}, _swapchain.extent), 1, 0, colorAttachment));

    _frameOpen = true;
}

void EngineVulkan::Clear(bool /*clearZ*/, bool clear, PackedColor color)
{
    // no depth attachment yet, ignore clearZ
    if (!clear)
        return;
    _clearColor = color;
    if (!_frameOpen)
        return;

    vk::ClearAttachment attachment(vk::ImageAspectFlagBits::eColor, 0, vk::ClearValue(ToClearColor(color)));
    vk::ClearRect rect(vk::Rect2D({0, 0}, _swapchain.extent), 0, 1);
    _frames[_frameIndex].cmd.clearAttachments(attachment, rect);
}

void EngineVulkan::FinishDraw()
{
    if (!_frameOpen)
        return;

    FrameResources& f = _frames[_frameIndex];
    f.cmd.endRendering();

    TransitionImage(f.cmd, _swapchain.images[_imageIndex], vk::ImageLayout::eColorAttachmentOptimal,
                    vk::ImageLayout::ePresentSrcKHR, vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                    vk::AccessFlagBits2::eColorAttachmentWrite, vk::PipelineStageFlagBits2::eNone, vk::AccessFlags2{});
    f.cmd.end();

    _frameOpen = false;
    _frameRecorded = true;
}

void EngineVulkan::NextFrame()
{
    if (!_frameRecorded)
        return;
    FrameResources& f = _frames[_frameIndex];

    vk::SemaphoreSubmitInfo waitAcquire(f.imageAvailable, 0, vk::PipelineStageFlagBits2::eColorAttachmentOutput);
    vk::CommandBufferSubmitInfo cmdInfo(f.cmd);
    vk::SemaphoreSubmitInfo signalRendered(_swapchain.renderFinished[_imageIndex], 0,
                                           vk::PipelineStageFlagBits2::eAllCommands);
    _vk.graphicsQueue.submit2(vk::SubmitInfo2({}, waitAcquire, cmdInfo, signalRendered), f.inFlight);

    try
    {
        const vk::Result r = _vk.presentQueue.presentKHR(
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
