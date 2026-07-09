#pragma once

#include "VulkanContext.hpp"

#include <vector>

struct SDL_Window;

namespace Poseidon
{

class VulkanSwapchain
{
  public:
    // Create() also rebuilds: the existing handle goes in as oldSwapchain.
    // Returns false when the window is minimized (0x0) or creation failed.
    bool Create(VulkanContext& ctx, SDL_Window* window, int swapInterval = 1);
    void Destroy(VulkanContext& ctx);

    bool IsValid() const { return static_cast<bool>(swapchain); }

    vk::SwapchainKHR swapchain;
    vk::Format format = vk::Format::eUndefined;
    vk::Extent2D extent;
    std::vector<vk::Image> images;
    std::vector<vk::ImageView> views;
    // Per image, not per frame slot: present waits on a semaphore tied to
    // the image, which can be re-acquired before its old slot cycles.
    std::vector<vk::Semaphore> renderFinished;
};

} // namespace Poseidon
