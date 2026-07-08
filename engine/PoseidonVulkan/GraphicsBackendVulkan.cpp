#include "EngineVulkan.hpp"
#include <Poseidon/Graphics/GraphicsEngineFactory.hpp>

using Poseidon::Engine;
using Poseidon::GraphicsBackendDescriptor;
using Poseidon::GraphicsEngineFactory;
using Poseidon::GraphicsEngineParams;

namespace
{
Engine* CreateVulkanBackend(const GraphicsEngineParams& params)
{
    return Poseidon::CreateEngineVulkan(params.width, params.height, params.useWindow, params.bitsPerPixel);
}

bool IsVulkanAvailable()
{
    return true;
}
} // namespace

namespace Poseidon
{
void RegisterVulkanGraphicsBackend()
{
    GraphicsEngineFactory::Register(GraphicsBackendDescriptor{
        "vk",
        "Vulkan 1.3 (SDL3)",
        90,
        &CreateVulkanBackend,
        &IsVulkanAvailable,
    });
}
} // namespace Poseidon
