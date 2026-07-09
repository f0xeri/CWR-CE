#include "VulkanShaderCompiler.hpp"

#include <Poseidon/Foundation/Logging/Logging.hpp>

#include <glslang/Public/ShaderLang.h>
#include <glslang/Public/ResourceLimits.h>
#include <glslang/SPIRV/GlslangToSpv.h>

namespace
{

EShLanguage ToGlslangStage(vk::ShaderStageFlagBits stage)
{
    switch (stage)
    {
        case vk::ShaderStageFlagBits::eVertex:
            return EShLangVertex;
        case vk::ShaderStageFlagBits::eFragment:
            return EShLangFragment;
        default:
            return EShLangCount;
    }
}

struct GlslangProcess
{
    GlslangProcess() { glslang::InitializeProcess(); }
    ~GlslangProcess() { glslang::FinalizeProcess(); }
};

} // namespace

namespace Poseidon
{

std::vector<uint32_t> CompileGlslToSpirv(vk::ShaderStageFlagBits stage, const char* source, const char* debugName)
{
    // InitializeProcess/FinalizeProcess should be called ONCE per proccess, so keep them static if we wan't to recompile something
    static GlslangProcess process;

    const EShLanguage lang = ToGlslangStage(stage);
    if (lang == EShLangCount)
    {
        LOG_ERROR(Graphics, "VK: unsupported shader stage for '{}'", debugName);
        return {};
    }

    glslang::TShader shader(lang);
    shader.setStrings(&source, 1);
    shader.setEnvInput(glslang::EShSourceGlsl, lang, glslang::EShClientVulkan, 100);
    shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_3);
    shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_6);

    if (!shader.parse(GetDefaultResources(), 450, false, EShMsgDefault))
    {
        LOG_ERROR(Graphics, "VK: GLSL compile failed for '{}':\n{}", debugName, shader.getInfoLog());
        return {};
    }

    glslang::TProgram program;
    program.addShader(&shader);
    if (!program.link(EShMsgDefault))
    {
        LOG_ERROR(Graphics, "VK: shader link failed for '{}':\n{}", debugName, program.getInfoLog());
        return {};
    }

    std::vector<uint32_t> spirv;
    glslang::GlslangToSpv(*program.getIntermediate(lang), spirv);
    if (spirv.empty())
        LOG_ERROR(Graphics, "VK: SPIR-V generation produced no code for '{}'", debugName);
    return spirv;
}

} // namespace Poseidon
