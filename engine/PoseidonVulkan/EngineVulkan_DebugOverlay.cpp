// ImGui dev overlay — Vulkan renderer impl.
//
// DebugOverlay (Poseidon/Dev/Debug) owns the ImGui context, the SDL3
// platform backend, the panel contents and the toggle hotkey; this file
// owns the ImGui_ImplVulkan_* renderer side, mirroring how the Metal
// backend drives its impl around DebugOverlay::NewFrame()/Render().
//
// The draw data is recorded into the frame's command buffer inside the
// open dynamic-rendering pass (FinishDraw, after the game queues flush,
// before endRendering) so the panel composites on top of everything.
#include "EngineVulkan.hpp"

#include <algorithm>

#include <Poseidon/Foundation/Logging/Logging.hpp>

// The PCH pulls in Logging.hpp which #defines DebugLog() as a logging macro.
// That collides with the method ImGui::DebugLog().  Undef before including
// ImGui headers — none of our code in this TU uses the DebugLog macro.
#ifdef DebugLog
#undef DebugLog
#endif

#include <imgui.h>
#include <imgui_impl_vulkan.h>

#include <Poseidon/Dev/Debug/DebugOverlay.hpp>

namespace
{

// ImGui_ImplVulkan_Init shallow-copies InitInfo; the color-format array it
// points to must outlive the backend.
VkFormat s_overlayColorFormat = VK_FORMAT_UNDEFINED;

void OverlayCheckVkResult(VkResult err)
{
    if (err != VK_SUCCESS)
        LOG_ERROR(Graphics, "VK: ImGui backend error: VkResult {}", (int)err);
}

} // namespace

namespace Poseidon
{

void EngineVulkan::InitDebugOverlay()
{
    Dev::DebugOverlay::InitForVulkan(_sdlWindow);
    if (!Dev::DebugOverlay::IsInitialized())
        return;

    s_overlayColorFormat = static_cast<VkFormat>(_swapchain.format);

    ImGui_ImplVulkan_InitInfo info = {};
    info.ApiVersion = VK_API_VERSION_1_3;
    info.Instance = _vk.instance;
    info.PhysicalDevice = _vk.physicalDevice;
    info.Device = _vk.device;
    info.QueueFamily = _vk.graphicsQueueFamily;
    info.Queue = _vk.graphicsQueue;
    info.DescriptorPoolSize = 64; // backend creates its own pool
    info.MinImageCount = 2;
    info.ImageCount = std::max<uint32_t>(2, (uint32_t)_swapchain.images.size());
    info.UseDynamicRendering = true;
    info.PipelineInfoMain.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
    info.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    info.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &s_overlayColorFormat;
    info.PipelineInfoMain.PipelineRenderingCreateInfo.depthAttachmentFormat = static_cast<VkFormat>(_depthFormat);
    info.PipelineInfoMain.PipelineRenderingCreateInfo.stencilAttachmentFormat = static_cast<VkFormat>(_depthFormat);
    info.CheckVkResultFn = OverlayCheckVkResult;

    if (!ImGui_ImplVulkan_Init(&info))
    {
        LOG_ERROR(Graphics, "VK: ImGui_ImplVulkan_Init failed — dev overlay disabled");
        Dev::DebugOverlay::Shutdown();
        return;
    }

    _imguiReady = true;
}

void EngineVulkan::ShutdownDebugOverlay()
{
    // Caller guarantees the device is idle (dtor waitIdle).
    if (_imguiReady)
    {
        ImGui_ImplVulkan_Shutdown();
        _imguiReady = false;
    }
    Dev::DebugOverlay::Shutdown();
}

void EngineVulkan::RenderDebugOverlay(vk::CommandBuffer cmd)
{
    if (!_imguiReady)
        return;
    ImGui_ImplVulkan_NewFrame();
    Dev::DebugOverlay::NewFrame();
    Dev::DebugOverlay::Render(); // builds draw data; GL branch no-ops for Vulkan
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), static_cast<VkCommandBuffer>(cmd));
    // The impl bound its own pipeline/scissor; our per-frame caches are reset
    // in the next InitDraw, and no game draws follow within this frame.
}

} // namespace Poseidon
