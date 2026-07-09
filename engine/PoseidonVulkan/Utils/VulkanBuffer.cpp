#include "VulkanBuffer.hpp"

#include <Poseidon/Foundation/Logging/Logging.hpp>

namespace Poseidon
{

bool VulkanBuffer::Create(VulkanContext& ctx, vk::DeviceSize cap, vk::BufferUsageFlags usage, const char* debugName)
{
    VkBufferCreateInfo bufferInfo = vk::BufferCreateInfo({}, cap, usage, vk::SharingMode::eExclusive);

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    // Coherent host memory: writes are visible to the GPU at submit without
    // explicit vmaFlushAllocation calls. Universally available on desktop.
    allocInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    VkBuffer rawBuffer = VK_NULL_HANDLE;
    VmaAllocationInfo outInfo{};
    if (vmaCreateBuffer(ctx.allocator, &bufferInfo, &allocInfo, &rawBuffer, &_alloc, &outInfo) != VK_SUCCESS)
    {
        LOG_ERROR(Graphics, "VK: ring buffer '{}' creation failed ({} bytes)", debugName, (uint64_t)cap);
        return false;
    }
    buffer = rawBuffer;
    mapped = static_cast<uint8_t*>(outInfo.pMappedData);
    capacity = cap;
    _cursor = 0;
    return true;
}

void VulkanBuffer::Destroy(VulkanContext& ctx)
{
    if (buffer)
        vmaDestroyBuffer(ctx.allocator, buffer, _alloc);
    buffer = nullptr;
    mapped = nullptr;
    _alloc = nullptr;
    capacity = 0;
    _cursor = 0;
}

void* VulkanBuffer::Allocate(vk::DeviceSize size, vk::DeviceSize alignment, vk::DeviceSize& outOffset)
{
    const vk::DeviceSize aligned = (alignment > 1) ? ((_cursor + alignment - 1) & ~(alignment - 1)) : _cursor;
    if (aligned + size > capacity)
        return nullptr;
    outOffset = aligned;
    _cursor = aligned + size;
    return mapped + aligned;
}

} // namespace Poseidon
