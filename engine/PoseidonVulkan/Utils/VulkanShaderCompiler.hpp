#pragma once

#include <vulkan/vulkan.hpp>

#include <vector>

namespace Poseidon
{

std::vector<uint32_t> CompileGlslToSpirv(vk::ShaderStageFlagBits stage, const char* source, const char* debugName);

} // namespace Poseidon
