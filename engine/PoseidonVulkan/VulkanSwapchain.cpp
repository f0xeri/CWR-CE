#include "VulkanSwapchain.hpp"

#include <Poseidon/Foundation/Logging/Logging.hpp>
#include <SDL3/SDL.h>

#include <algorithm>

namespace
{

// unorm, not sRGB: write semantics must match the GL33 default framebuffer.
vk::SurfaceFormatKHR ChooseSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& formats)
{
    for (const auto& f : formats)
        if (f.format == vk::Format::eB8G8R8A8Unorm && f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
            return f;
    for (const auto& f : formats)
        if (f.format == vk::Format::eR8G8B8A8Unorm && f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
            return f;
    return formats.front();
}

// GL swap-interval semantics: 1 = vsync (FIFO, guaranteed everywhere),
// 0 = uncapped (IMMEDIATE, falling back to MAILBOX which is uncapped but
// tear-free), -1 = adaptive (FIFO_RELAXED).
vk::PresentModeKHR ChoosePresentMode(const std::vector<vk::PresentModeKHR>& modes, int swapInterval)
{
    auto has = [&](vk::PresentModeKHR m) { return std::find(modes.begin(), modes.end(), m) != modes.end(); };
    if (swapInterval == 0)
    {
        if (has(vk::PresentModeKHR::eImmediate))
            return vk::PresentModeKHR::eImmediate;
        if (has(vk::PresentModeKHR::eMailbox))
            return vk::PresentModeKHR::eMailbox;
    }
    else if (swapInterval < 0 && has(vk::PresentModeKHR::eFifoRelaxed))
    {
        return vk::PresentModeKHR::eFifoRelaxed;
    }
    return vk::PresentModeKHR::eFifo;
}

vk::Extent2D ChooseExtent(const vk::SurfaceCapabilitiesKHR& caps, SDL_Window* window)
{
    if (caps.currentExtent.width != 0xFFFFFFFFu)
        return caps.currentExtent;
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(window, &w, &h);
    return {std::clamp((uint32_t)w, caps.minImageExtent.width, caps.maxImageExtent.width),
            std::clamp((uint32_t)h, caps.minImageExtent.height, caps.maxImageExtent.height)};
}

} // namespace

namespace Poseidon
{

bool VulkanSwapchain::Create(VulkanContext& ctx, SDL_Window* window, int swapInterval)
{
    try
    {
        const auto caps = ctx.physicalDevice.getSurfaceCapabilitiesKHR(ctx.surface);
        const auto surfaceFormat = ChooseSurfaceFormat(ctx.physicalDevice.getSurfaceFormatsKHR(ctx.surface));
        const auto presentMode =
            ChoosePresentMode(ctx.physicalDevice.getSurfacePresentModesKHR(ctx.surface), swapInterval);
        const vk::Extent2D newExtent = ChooseExtent(caps, window);
        if (newExtent.width == 0 || newExtent.height == 0)
            return false; // minimized — caller retries later

        // min+1 so acquire doesn't block on the bare minimum; max 0 = no limit
        uint32_t imageCount = caps.minImageCount + 1;
        if (caps.maxImageCount > 0)
            imageCount = std::min(imageCount, caps.maxImageCount);

        vk::SwapchainCreateInfoKHR ci(
            {}, ctx.surface, imageCount, surfaceFormat.format, surfaceFormat.colorSpace, newExtent,
            /*imageArrayLayers=*/1, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst,
            vk::SharingMode::eExclusive, nullptr, caps.currentTransform, vk::CompositeAlphaFlagBitsKHR::eOpaque,
            presentMode, /*clipped=*/true, /*oldSwapchain=*/swapchain);

        vk::SwapchainKHR newSwapchain = ctx.device.createSwapchainKHR(ci);

        // Old swapchain was retired via oldSwapchain; caller guarantees the
        // GPU is idle, so its views/semaphores can be freed now.
        Destroy(ctx);
        swapchain = newSwapchain;
        format = surfaceFormat.format;
        extent = newExtent;

        images = ctx.device.getSwapchainImagesKHR(swapchain);
        views.reserve(images.size());
        renderFinished.reserve(images.size());
        for (const auto& image : images)
        {
            views.push_back(ctx.device.createImageView(
                vk::ImageViewCreateInfo({}, image, vk::ImageViewType::e2D, format, {},
                                        vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1))));
            renderFinished.push_back(ctx.device.createSemaphore({}));
        }

        LOG_INFO(Graphics, "VK: swapchain {}x{}, {} images, {}, {}", extent.width, extent.height, images.size(),
                 vk::to_string(format), vk::to_string(presentMode));
        return true;
    }
    catch (const std::exception& e)
    {
        LOG_ERROR(Graphics, "VK: swapchain creation failed: {}", e.what());
        return false;
    }
}

void VulkanSwapchain::Destroy(VulkanContext& ctx)
{
    for (auto view : views)
        ctx.device.destroyImageView(view);
    views.clear();
    for (auto sem : renderFinished)
        ctx.device.destroySemaphore(sem);
    renderFinished.clear();
    images.clear();
    if (swapchain)
    {
        ctx.device.destroySwapchainKHR(swapchain);
        swapchain = nullptr;
    }
    format = vk::Format::eUndefined;
    extent = vk::Extent2D{};
}

} // namespace Poseidon
