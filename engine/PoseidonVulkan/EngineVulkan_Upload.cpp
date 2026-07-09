// Texture-upload service: staging arena + per-frame upload command buffer +
// deferred destruction. GL33 uploads mid-frame with glTexSubImage2D executing
// in call order; here copy commands can't be recorded inside the frame's
// active dynamic-rendering pass, so they collect in a second command buffer
// submitted BEFORE the frame's draws. The only observable difference: a draw
// recorded before a same-frame re-upload samples the new bytes instead of the
// old ones — a one-frame, better-mip-earlier artifact.
#include "EngineVulkan.hpp"

#include <Poseidon/Foundation/Logging/Logging.hpp>

namespace Poseidon
{

EngineVulkan::UploadTicket EngineVulkan::BeginTextureUpload()
{
    UploadTicket ticket;
    if (_frameOpen)
    {
        FrameResources& f = _frames[_frameIndex];
        if (!_uploadOpen)
        {
            f.uploadCmd.reset();
            f.uploadCmd.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
            _uploadOpen = true;
        }
        ticket.cmd = f.uploadCmd;
        ticket.immediate = false;
        return ticket;
    }

    // No frame open (loading screen, preload): blocking one-shot submit.
    ticket.immediate = true;
    ticket.tempPool = _vk.device.createCommandPool(
        vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eTransient, _vk.graphicsQueueFamily));
    ticket.cmd =
        _vk.device.allocateCommandBuffers({ticket.tempPool, vk::CommandBufferLevel::ePrimary, 1}).front();
    ticket.cmd.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    return ticket;
}

void EngineVulkan::EndTextureUpload(UploadTicket& ticket)
{
    if (!ticket.immediate)
    {
        // Frame path: uploadCmd stays open collecting further uploads; it is
        // ended and submitted in NextFrame. Overflow staging buffers must
        // survive until that submit retires.
        for (VulkanBuffer& b : ticket.tempBuffers)
            _deferredBuffers[DeferSlot()].push_back(b);
        ticket.tempBuffers.clear();
        return;
    }

    ticket.cmd.end();
    vk::CommandBufferSubmitInfo cmdInfo(ticket.cmd);
    _vk.graphicsQueue.submit2(vk::SubmitInfo2({}, {}, cmdInfo), nullptr);
    _vk.graphicsQueue.waitIdle();
    _vk.device.destroyCommandPool(ticket.tempPool);
    ticket.tempPool = nullptr;
    for (VulkanBuffer& b : ticket.tempBuffers)
        b.Destroy(_vk);
    ticket.tempBuffers.clear();
}

uint8_t* EngineVulkan::AllocStaging(UploadTicket& ticket, vk::DeviceSize size, vk::Buffer& outBuffer,
                                    vk::DeviceSize& outOffset)
{
    if (!ticket.immediate)
    {
        VulkanBuffer& arena = _stagingBuffer[_frameIndex];
        if (void* p = arena.Allocate(size, 16, outOffset))
        {
            outBuffer = arena.buffer;
            return static_cast<uint8_t*>(p);
        }
    }

    // Arena exhausted or immediate path: dedicated one-shot staging buffer.
    VulkanBuffer temp;
    if (!temp.Create(_vk, size, vk::BufferUsageFlagBits::eTransferSrc, "tex-staging-temp"))
        return nullptr;
    ticket.tempBuffers.push_back(temp);
    outBuffer = temp.buffer;
    outOffset = 0;
    return ticket.tempBuffers.back().mapped;
}

int EngineVulkan::DeferSlot() const
{
    // While a frame is recording (or recorded but not yet submitted), this
    // slot's fence is the last to signal, so its flush point covers everything.
    // Between frames _frameIndex already points at the NEXT slot, whose fence
    // retired long ago — the previous slot's fence covers the newest submit,
    // which may still reference the resource through its descriptor sets.
    if (_frameOpen || _frameRecorded)
        return _frameIndex;
    return (_frameIndex + kFramesInFlight - 1) % kFramesInFlight;
}

void EngineVulkan::DeferDestroyImage(vk::Image image, VmaAllocation alloc, vk::ImageView view)
{
    if (!image && !view)
        return;
    _deferredImages[DeferSlot()].push_back({image, alloc, view});
}

void EngineVulkan::DeferDestroyBuffer(vk::Buffer buffer, VmaAllocation alloc)
{
    if (!buffer)
        return;
    _deferredRawBuffers[DeferSlot()].push_back({buffer, alloc});
}

void EngineVulkan::FlushDeferredDestroys(int slot)
{
    for (const DeferredImage& d : _deferredImages[slot])
    {
        if (d.view)
            _vk.device.destroyImageView(d.view);
        if (d.image)
            vmaDestroyImage(_vk.allocator, d.image, d.alloc);
    }
    _deferredImages[slot].clear();
    for (VulkanBuffer& b : _deferredBuffers[slot])
        b.Destroy(_vk);
    _deferredBuffers[slot].clear();
    for (const DeferredRawBuffer& b : _deferredRawBuffers[slot])
        vmaDestroyBuffer(_vk.allocator, b.buffer, b.alloc);
    _deferredRawBuffers[slot].clear();
}

void EngineVulkan::FlushAllDeferredDestroys()
{
    for (int i = 0; i < kFramesInFlight; ++i)
        FlushDeferredDestroys(i);
}

} // namespace Poseidon
