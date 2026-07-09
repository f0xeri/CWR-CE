#pragma once

#include "../VulkanContext.hpp"

namespace Poseidon
{

class VulkanBuffer
{
  public:
    bool Create(VulkanContext& ctx, vk::DeviceSize capacity, vk::BufferUsageFlags usage, const char* debugName);
    void Destroy(VulkanContext& ctx);

    void Reset() { _cursor = 0; }

    // Linear-allocates `size` bytes at `alignment`; returns the write pointer
    // and the buffer offset, or nullptr when the frame's budget is exhausted.
    void* Allocate(vk::DeviceSize size, vk::DeviceSize alignment, vk::DeviceSize& outOffset);

    vk::Buffer buffer;
    uint8_t* mapped = nullptr;
    vk::DeviceSize capacity = 0;

  private:
    VmaAllocation _alloc = nullptr;
    vk::DeviceSize _cursor = 0;
};

} // namespace Poseidon
