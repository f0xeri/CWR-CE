#pragma once

#include <vulkan/vulkan.hpp>

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>

struct SDL_Window;

namespace Poseidon
{

inline constexpr uint32_t kNoQueueFamily = VK_QUEUE_FAMILY_IGNORED;

class VulkanContext
{
  public:
    bool Init(SDL_Window* window);
    void Shutdown();

    bool IsValid() const { return device != nullptr; }

    vk::Instance instance;
    vk::DebugUtilsMessengerEXT debugMessenger; // null when validation layer is absent
    vk::SurfaceKHR surface;
    vk::PhysicalDevice physicalDevice;
    vk::PhysicalDeviceProperties deviceProps;
    vk::Device device;

    vk::Queue graphicsQueue;
    vk::Queue presentQueue;
    uint32_t graphicsQueueFamily = kNoQueueFamily;
    uint32_t presentQueueFamily = kNoQueueFamily;

    VmaAllocator allocator = nullptr;
};

} // namespace Poseidon
