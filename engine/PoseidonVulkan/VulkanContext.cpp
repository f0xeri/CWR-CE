// The one TU that hosts the VMA implementation and the storage for the
// Vulkan-Hpp default dispatcher (the global table of function pointers that
// every vk:: call routes through).
#define VMA_IMPLEMENTATION
#include "VulkanContext.hpp"

#include <Poseidon/Foundation/Logging/Logging.hpp>
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <vector>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace
{
using Poseidon::kNoQueueFamily;

constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";

VKAPI_ATTR VkBool32 VKAPI_CALL VkDebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                               VkDebugUtilsMessageTypeFlagsEXT /*types*/,
                                               const VkDebugUtilsMessengerCallbackDataEXT* data, void* /*user*/)
{
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        LOG_ERROR(Graphics, "VK validation: {}", data->pMessage);
    else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        LOG_WARN(Graphics, "VK validation: {}", data->pMessage);
    else
        LOG_INFO(Graphics, "VK validation: {}", data->pMessage);

    return VK_FALSE;
}

bool HasLayer(const char* name)
{
    for (const auto& lp : vk::enumerateInstanceLayerProperties())
        if (strcmp(lp.layerName, name) == 0)
            return true;
    return false;
}

uint32_t FindCombinedQueueFamily(vk::PhysicalDevice dev, vk::SurfaceKHR surface)
{
    const auto families = dev.getQueueFamilyProperties2();
    for (uint32_t i = 0; i < (uint32_t)families.size(); ++i)
    {
        if ((families[i].queueFamilyProperties.queueFlags & vk::QueueFlagBits::eGraphics) && dev.getSurfaceSupportKHR(i, surface))
            return i;
    }
    return kNoQueueFamily;
}

bool SupportsRequiredFeatures(vk::PhysicalDevice dev)
{
    if (dev.getProperties().apiVersion < VK_API_VERSION_1_3)
        return false;

    bool hasSwapchainExt = false;
    for (const auto& ext : dev.enumerateDeviceExtensionProperties())
        if (strcmp(ext.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0)
            hasSwapchainExt = true;
    if (!hasSwapchainExt)
        return false;

    auto chain = dev.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features>();
    const auto& f13 = chain.get<vk::PhysicalDeviceVulkan13Features>();
    return f13.dynamicRendering && f13.synchronization2;
}

bool InitDispatcher()
{
    VULKAN_HPP_DEFAULT_DISPATCHER.init();

    const uint32_t loaderVersion = vk::enumerateInstanceVersion();
    if (loaderVersion < VK_API_VERSION_1_3)
    {
        LOG_WARN(Graphics, "VK: loader only supports {}.{}, need 1.3 — backend unavailable",
                 VK_API_VERSION_MAJOR(loaderVersion), VK_API_VERSION_MINOR(loaderVersion));
        return false;
    }
    return true;
}

vk::Instance CreateInstance(bool wantValidation)
{
    Uint32 sdlExtCount = 0;
    const char* const* sdlExts = SDL_Vulkan_GetInstanceExtensions(&sdlExtCount);
    if (!sdlExts)
    {
        LOG_ERROR(Graphics, "VK: SDL_Vulkan_GetInstanceExtensions failed: {}", SDL_GetError());
        return nullptr;
    }
    std::vector<const char*> instanceExts(sdlExts, sdlExts + sdlExtCount);
    std::vector<const char*> layers;
    if (wantValidation)
    {
        layers.push_back(kValidationLayer);
        instanceExts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    vk::ApplicationInfo appInfo("CWR", 1, "Poseidon", 1, VK_API_VERSION_1_3);
    vk::Instance instance = vk::createInstance(vk::InstanceCreateInfo({}, &appInfo, layers, instanceExts));
    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance); // now instance-level functions are loaded
    return instance;
}

vk::DebugUtilsMessengerEXT CreateDebugMessenger(vk::Instance instance)
{
    using Sev = vk::DebugUtilsMessageSeverityFlagBitsEXT;
    using Type = vk::DebugUtilsMessageTypeFlagBitsEXT;
    return instance.createDebugUtilsMessengerEXT(vk::DebugUtilsMessengerCreateInfoEXT(
        {}, Sev::eError | Sev::eWarning, Type::eValidation | Type::ePerformance | Type::eGeneral, &VkDebugCallback));
}

vk::SurfaceKHR CreateWindowSurface(SDL_Window* window, vk::Instance instance)
{
    VkSurfaceKHR rawSurface = VK_NULL_HANDLE;
    if (!SDL_Vulkan_CreateSurface(window, instance, nullptr, &rawSurface))
    {
        LOG_ERROR(Graphics, "VK: SDL_Vulkan_CreateSurface failed: {}", SDL_GetError());
        return nullptr;
    }
    return rawSurface;
}

struct GpuPick
{
    vk::PhysicalDevice device;
    uint32_t queueFamily = kNoQueueFamily;
};

GpuPick PickPhysicalDevice(vk::Instance instance, vk::SurfaceKHR surface)
{
    GpuPick best;
    int bestScore = -1;
    for (const auto& dev : instance.enumeratePhysicalDevices())
    {
        if (!SupportsRequiredFeatures(dev))
            continue;
        const uint32_t family = FindCombinedQueueFamily(dev, surface);
        if (family == kNoQueueFamily)
            continue;
        const int score = (dev.getProperties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu) ? 1000 : 100;
        if (score > bestScore)
        {
            bestScore = score;
            best.device = dev;
            best.queueFamily = family;
        }
    }
    return best;
}

vk::Device CreateLogicalDevice(vk::PhysicalDevice physicalDevice, uint32_t queueFamily)
{
    const float queuePriority = 1.0f;
    const vk::DeviceQueueCreateInfo queueInfo({}, queueFamily, 1, &queuePriority);
    const char* deviceExts[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    vk::PhysicalDeviceVulkan13Features features13;
    features13.dynamicRendering = true;
    features13.synchronization2 = true;
    vk::PhysicalDeviceFeatures baseFeatures;
    baseFeatures.samplerAnisotropy = physicalDevice.getFeatures().samplerAnisotropy;

    vk::DeviceCreateInfo deviceInfo({}, 1, &queueInfo, 0, nullptr, 1, deviceExts, &baseFeatures);
    deviceInfo.pNext = &features13;
    vk::Device device = physicalDevice.createDevice(deviceInfo);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(device); // device-level functions, bypassing loader trampolines
    return device;
}

VmaAllocator CreateAllocator(vk::Instance instance, vk::PhysicalDevice physicalDevice, vk::Device device)
{
    VmaVulkanFunctions vmaFns{};
    vmaFns.vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr;
    vmaFns.vkGetDeviceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;
    VmaAllocatorCreateInfo allocatorInfo{};
    allocatorInfo.instance = instance;
    allocatorInfo.physicalDevice = physicalDevice;
    allocatorInfo.device = device;
    allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;
    allocatorInfo.pVulkanFunctions = &vmaFns;

    VmaAllocator allocator = nullptr;
    if (vmaCreateAllocator(&allocatorInfo, &allocator) != VK_SUCCESS)
    {
        LOG_ERROR(Graphics, "VK: vmaCreateAllocator failed");
        return nullptr;
    }
    return allocator;
}

} // namespace

namespace Poseidon
{

bool VulkanContext::Init(SDL_Window* window)
{
    try
    {
        if (!InitDispatcher())
            return false;

        // Validation costs a large chunk of frame time — opt-in only.
        // Set CWR_VK_VALIDATION=1 to enable (requires the Vulkan SDK layer).
        const char* envValidation = std::getenv("CWR_VK_VALIDATION");
        const bool wantValidation = envValidation && *envValidation == '1' && HasLayer(kValidationLayer);
        instance = CreateInstance(wantValidation);
        if (!instance)
            return false;
        if (wantValidation)
            debugMessenger = CreateDebugMessenger(instance);
        LOG_INFO(Graphics, "VK: instance created (validation: {})", wantValidation ? "on" : "off");

        surface = CreateWindowSurface(window, instance);
        if (!surface)
            return false;

        const GpuPick pick = PickPhysicalDevice(instance, surface);
        if (!pick.device)
        {
            LOG_WARN(Graphics,"VK: no Vulkan 1.3 device with dynamic rendering and combined graphics+present queue - backend unavailable");
            return false;
        }
        physicalDevice = pick.device;
        graphicsQueueFamily = pick.queueFamily;
        deviceProps = physicalDevice.getProperties();
        LOG_INFO(Graphics, "VK: using {} (Vulkan {}.{}.{}, {})", (const char*)deviceProps.deviceName,
                 VK_API_VERSION_MAJOR(deviceProps.apiVersion), VK_API_VERSION_MINOR(deviceProps.apiVersion),
                 VK_API_VERSION_PATCH(deviceProps.apiVersion), vk::to_string(deviceProps.deviceType));

        device = CreateLogicalDevice(physicalDevice, graphicsQueueFamily);
        graphicsQueue = device.getQueue(graphicsQueueFamily, 0);
        LOG_INFO(Graphics, "VK: combined graphics+present queue family {}", graphicsQueueFamily);

        allocator = CreateAllocator(instance, physicalDevice, device);
        if (!allocator)
        {
            Shutdown();
            return false;
        }

        return true;
    }
    catch (const std::exception& e)
    {
        LOG_ERROR(Graphics, "VK: initialization failed: {}", e.what());
        Shutdown();
        return false;
    }
}

void VulkanContext::Shutdown()
{
    if (device)
        device.waitIdle();
    if (allocator)
    {
        vmaDestroyAllocator(allocator);
        allocator = nullptr;
    }
    if (device)
    {
        device.destroy();
        device = nullptr;
    }
    if (surface)
    {
        instance.destroySurfaceKHR(surface);
        surface = nullptr;
    }
    if (debugMessenger)
    {
        instance.destroyDebugUtilsMessengerEXT(debugMessenger);
        debugMessenger = nullptr;
    }
    if (instance)
    {
        instance.destroy();
        instance = nullptr;
    }
    physicalDevice = nullptr;
    graphicsQueue = nullptr;
    graphicsQueueFamily = kNoQueueFamily;
}

} // namespace Poseidon
