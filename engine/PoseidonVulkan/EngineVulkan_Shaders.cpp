// GLSL 450 ports of the GL33 screen-path shaders (EngineGL33_Shaders.cpp) and
// the CPU-side constant machinery. The std140 block layouts are byte-identical
// to GL33's VSConst / PSConstants slot maps so producer-visible semantics stay
// 1:1; only the resource addressing (set/binding, in/out locations) is Vulkan.
//
// Phase-1 scope: VSScreen + PSNormal (day, no cascade shadow sampling — the
// shadowCtl path is never enabled for screen-space draws) + PSFlat.
#include "EngineVulkan.hpp"
#include "Utils/VulkanShaderCompiler.hpp"

#include <Poseidon/Foundation/Logging/Logging.hpp>

namespace
{

// Slot map mirror of PoseidonGL33's VSConst / PSConstants::Slot enums.
namespace slots
{
enum : int
{
    VSFogParam = 16,
    VSVpScale = 21,

    PSFogColor = 0,
    PSAlphaRef = 1,
    PSConstColor = 3,
    PSRgbEyeCoef = 7,
};
} // namespace slots

const char* const kVSScreen = R"(#version 450
// Shared VS UBO; vsScreen reads vpScale at slot 21 (offset 336 bytes). The
// prefix is declared for std140 layout parity with vsTransform (see GL33).
layout(set = 0, binding = 0, std140) uniform VSConstants {
    mat4 _pad_proj;     // slots 0..3
    mat4 _pad_view;     // 4..7
    mat4 _pad_world;    // 8..11
    vec4 _pad_sunDir;   // 12
    vec4 _pad_ambient;  // 13
    vec4 _pad_diffuse;  // 14
    vec4 _pad_emissive; // 15
    vec4 _pad_fog;      // 16
    vec4 _pad_camPos;   // 17
    vec4 _pad_spec;     // 18
    vec4 _pad_specEn;   // 19
    vec4 _pad_sunEn;    // 20
    vec4 vpScale;       // 21 — {2/width, 2/height, 0, 0}
};

layout(location = 0) in vec3 aPos;
layout(location = 1) in float aRhw;
layout(location = 2) in vec4 aColor;
layout(location = 3) in vec4 aSpecular;
layout(location = 4) in vec2 aUV0;
layout(location = 5) in vec2 aUV1;

layout(location = 0) out vec4 vColor;
layout(location = 1) out vec4 vSpecColor;
layout(location = 2) out vec2 vUV0;
layout(location = 3) out vec2 vUV1;
layout(location = 4) out float vFogTC;

void main() {
    // Identical math to GL33 (GL-style NDC, Y up); the record path uses a
    // negative-height viewport so rasterization lands like GL (doc 4.6).
    float w = 1.0 / aRhw;
    gl_Position.x = (aPos.x * vpScale.x - 1.0) * w;
    gl_Position.y = (1.0 - aPos.y * vpScale.y) * w;
    gl_Position.z = aPos.z * w;
    gl_Position.w = w;
    vColor = aColor;
    vSpecColor = aSpecular;
    vUV0 = aUV0;
    vUV1 = aUV1;
    vFogTC = aSpecular.a;
}
)";

const char* const kPSNormal = R"(#version 450
layout(set = 0, binding = 1, std140) uniform PSConstants {
    vec4 fogColor;    // c0
    vec4 alphaRef;    // c1: {ref, enabled, alphaToCoverage, flatDebug}
    vec4 shadowCtl;   // c2 (unused in phase 1 — screen draws never enable it)
    vec4 constColor;  // c3: per-object IsColored tint (white = no-op)
    vec4 _pad4;
    vec4 _pad5;
    vec4 _pad6;
    vec4 rgbEyeCoef;  // c7
};

layout(set = 0, binding = 2) uniform sampler2D tex0;

layout(location = 0) in vec4 vColor;
layout(location = 1) in vec4 vSpecColor;
layout(location = 2) in vec2 vUV0;
layout(location = 3) in vec2 vUV1;
layout(location = 4) in float vFogTC;

layout(location = 0) out vec4 fragColor;

void main() {
    vec4 r0 = vColor * texture(tex0, vUV0);
    r0 *= constColor;
    r0.rgb += vSpecColor.rgb;

    if (alphaRef.z > 0.5) {
        float cov = clamp((r0.a - alphaRef.x) / max(fwidth(r0.a), 1e-4) + 0.5, 0.0, 1.0);
        if (cov <= 0.0) discard;
        r0.a = cov;
    } else if (r0.a - alphaRef.x * alphaRef.y < 0.0) discard;

    float luminance = clamp(dot(r0.rgb, rgbEyeCoef.rgb), 0.0, 1.0);
    float nightBlend = clamp(luminance + rgbEyeCoef.a, 0.0, 1.0);
    r0.rgb = mix(vec3(luminance), r0.rgb, nightBlend);

    r0.rgb = mix(fogColor.rgb, r0.rgb, vFogTC);
    fragColor = alphaRef.w > 0.5 ? vec4(1.0, 0.0, 0.0, 1.0) : r0;
}
)";

const char* const kPSFlat = R"(#version 450
layout(location = 0) in vec4 vColor;
layout(location = 1) in vec4 vSpecColor;
layout(location = 2) in vec2 vUV0;
layout(location = 3) in vec2 vUV1;
layout(location = 4) in float vFogTC;

layout(location = 0) out vec4 fragColor;

void main() {
    fragColor = vColor;
}
)";

// Shared VS UBO declaration for the mesh shaders — byte-identical std140
// layout to GL33's VSConstants (70 vec4 slots). The world matrix is read from
// the block itself (slots 8..11): the VK path snapshots the whole block per
// draw into the UBO ring; instanced runs read per-instance worlds from the
// WorldInstances block instead (see VK_WORLD_PUSH_CONSTANT below).
#define VK_VS_CONSTANTS_BLOCK \
    "layout(set = 0, binding = 0, std140) uniform VSConstants {\n" \
    "    mat4 proj;          // c0-c3\n" \
    "    mat4 view;          // c4-c7\n" \
    "    mat4 world;         // c8-c11\n" \
    "    vec4 sunDir;        // c12\n" \
    "    vec4 ambient;       // c13\n" \
    "    vec4 diffuse;       // c14\n" \
    "    vec4 emissive;      // c15\n" \
    "    vec4 fogParam;      // c16: {start, invRange, enabled, 0}\n" \
    "    vec4 camPos;        // c17\n" \
    "    vec4 specular;      // c18: rgb + power(w)\n" \
    "    vec4 specEn;        // c19: {enabled, 0, 0, 0}\n" \
    "    vec4 sunEn;         // c20: {enabled, 0, 0, 0}\n" \
    "    vec4 vpScale;       // c21 — VSScreen only, layout parity\n" \
    "    vec4 _pad22;\n" \
    "    vec4 _pad23;\n" \
    "    mat4 texMat0;       // c24-c27\n" \
    "    mat4 texMat1;       // c28-c31\n" \
    "    vec4 texCtrl;       // c32: {genTex0, genTex1, 0, 0}\n" \
    "    vec4 lightCount;    // c33: x = active local light count\n" \
    "    vec4 lightPos[8];   // c34-c41: xyz camera-relative pos, w = startAtten\n" \
    "    vec4 lightDiffuse[8];  // c42-c49\n" \
    "    vec4 lightAmbient[8];  // c50-c57\n" \
    "    vec4 localLightDir[8]; // c58-c65: xyz beam dir, w = isSpot\n" \
    "    mat4 lightVP;       // c66-c69\n" \
    "};\n"

// Per-draw world matrix as a push constant: GL33 updates a 64-byte UBO
// subrange per draw; push constants are the Vulkan analogue and let draws
// with unchanged materials reuse the previous descriptor set entirely.
// flags.x > 0.5 = instanced run: the world matrix comes from the
// WorldInstances UBO slice (binding 4) indexed by gl_InstanceID instead.
// The `world` member of VSConstants stays as std140 padding.
#define VK_WORLD_PUSH_CONSTANT \
    "layout(push_constant) uniform PushWorld { mat4 world; vec4 flags; } pc;\n" \
    "layout(set = 0, binding = 4, std140) uniform WorldInstances { mat4 worldArr[256]; };\n" \
    "mat4 FetchWorld() { return (pc.flags.x > 0.5) ? worldArr[gl_InstanceIndex] : pc.world; }\n"

// 3D mesh vertex shader — port of GL33's vsTransform (lighting, fog, texgen,
// local point/spot lights). Camera-relative: world matrix translation already
// has cameraPos subtracted, camPos in the block is zeroed.
const char* const kVSTransform = "#version 450\n" VK_VS_CONSTANTS_BLOCK VK_WORLD_PUSH_CONSTANT R"(
layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

layout(location = 0) out vec4 vColor;
layout(location = 1) out vec4 vSpecColor;
layout(location = 2) out vec2 vUV0;
layout(location = 3) out vec2 vUV1;
layout(location = 4) out float vFogTC;

void main() {
    mat4 worldM      = FetchWorld();
    vec4 worldPos    = worldM * vec4(pos, 1.0);
    vec3 worldNormal = normalize(mat3(worldM) * normal);
    vec4 viewPos     = view * worldPos;
    gl_Position      = proj * viewPos;

    float NdotL = max(0.0, dot(worldNormal, -sunDir.xyz));
    vec4 litColor;
    litColor.rgb = emissive.rgb + ambient.rgb * sunEn.x + diffuse.rgb * NdotL * sunEn.x;
    litColor.a   = emissive.a   + ambient.a   * sunEn.x + diffuse.a   * NdotL * sunEn.x;

    // Local lights (street lamps, headlights) — see GL33 vsTransform for the
    // legacy LightPoint/LightReflector derivation this mirrors.
    const float MIN_INSIDE2 = 0.95677279; // (cos 12deg)^2
    const float MAX_INSIDE2 = 0.98063081; // (cos 8deg)^2
    int nLights = int(lightCount.x);
    for (int i = 0; i < nLights; i++)
    {
        vec3 toLight = lightPos[i].xyz - worldPos.xyz;
        float size2 = dot(toLight, toLight);
        float startAtten2 = lightPos[i].w * lightPos[i].w;
        float endAtten2 = startAtten2 * 100.0;
        if (size2 >= endAtten2)
            continue;

        float cone = 1.0;
        if (localLightDir[i].w > 0.5)
        {
            float inside = -dot(toLight, localLightDir[i].xyz);
            if (inside <= 0.0)
                continue;
            float cos2 = (inside * inside) / size2;
            if (cos2 < MIN_INSIDE2)
                continue;
            cone = clamp((cos2 - MIN_INSIDE2) / (MAX_INSIDE2 - MIN_INSIDE2), 0.0, 1.0);
        }

        float atten = (size2 >= startAtten2) ? (startAtten2 / size2) : 1.0;
        float cosFi = dot(toLight, worldNormal);
        vec3 contrib;
        if (cosFi > 0.0)
        {
            cosFi *= inversesqrt(size2);
            contrib = (lightDiffuse[i].rgb * cosFi + lightAmbient[i].rgb) * (atten * cone);
        }
        else
        {
            contrib = lightAmbient[i].rgb * atten;
        }
        litColor.rgb += contrib;
    }

    vColor = clamp(litColor, 0.0, 1.0);

    vec3 spec = vec3(0.0);
    if (specEn.x > 0.5 && sunEn.x > 0.0) {
        vec3 viewDir = normalize(camPos.xyz - worldPos.xyz);
        vec3 halfVec = normalize(-sunDir.xyz + viewDir);
        float NdotH = max(0.0, dot(worldNormal, halfVec));
        float specPow = max(1.0, specular.w);
        spec = specular.rgb * pow(NdotH, specPow) * sunEn.x;
    }
    vSpecColor = vec4(clamp(spec, 0.0, 1.0), 0.0);

    float dist = length(worldPos.xyz - camPos.xyz);
    float fogFactor = clamp(1.0 - (dist - fogParam.x) * fogParam.y, 0.0, 1.0);
    vFogTC = (fogParam.z > 0.5) ? fogFactor : 1.0;

    vUV0 = (texCtrl.x > 0.5) ? (texMat0 * vec4(uv, 0, 1)).xy : uv;
    vUV1 = (texCtrl.y > 0.5) ? (texMat1 * vec4(uv, 0, 1)).xy : uv;
}
)";

// VSShadow — unlit transform for per-poly shadow draws (GL33 mirror).
const char* const kVSShadow = "#version 450\n" VK_VS_CONSTANTS_BLOCK VK_WORLD_PUSH_CONSTANT R"(
layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

layout(location = 0) out vec4 vColor;
layout(location = 1) out vec4 vSpecColor;
layout(location = 2) out vec2 vUV0;
layout(location = 3) out vec2 vUV1;
layout(location = 4) out float vFogTC;

void main() {
    vec4 worldPos = FetchWorld() * vec4(pos, 1.0);
    gl_Position   = proj * view * worldPos;
    vColor        = diffuse;   // unlit — direct from material.diffuse
    vSpecColor    = vec4(0.0);
    vUV0          = (texCtrl.x > 0.5) ? (texMat0 * vec4(uv, 0, 1)).xy : uv;
    vUV1          = vUV0;
    vFogTC        = 1.0;       // shadows ignore fog
}
)";

// Shared PS UBO declaration — GL33 PSConstants layout (27 vec4 slots). The
// cascade shadow-map tail (c8..c26) is declared for layout parity but never
// read: shadowCtl stays 0 until shadow maps land (phase 3).
#define VK_PS_CONSTANTS_BLOCK \
    "layout(set = 0, binding = 1, std140) uniform PSConstants {\n" \
    "    vec4 fogColor;    // c0\n" \
    "    vec4 alphaRef;    // c1: {ref, enabled, alphaToCoverage, flatDebug}\n" \
    "    vec4 shadowCtl;   // c2\n" \
    "    vec4 constColor;  // c3: per-object IsColored tint\n" \
    "    vec4 lightDir;    // c4 (water)\n" \
    "    vec4 grassCoef1;  // c5\n" \
    "    vec4 grassCoef2;  // c6\n" \
    "    vec4 rgbEyeCoef;  // c7\n" \
    "};\n"

// PSDetail — diffuse * tex0, detail modulation from tex1 alpha.
const char* const kPSDetail = "#version 450\n" VK_PS_CONSTANTS_BLOCK R"(
layout(set = 0, binding = 2) uniform sampler2D tex0;
layout(set = 0, binding = 3) uniform sampler2D tex1;

layout(location = 0) in vec4 vColor;
layout(location = 1) in vec4 vSpecColor;
layout(location = 2) in vec2 vUV0;
layout(location = 3) in vec2 vUV1;
layout(location = 4) in float vFogTC;

layout(location = 0) out vec4 fragColor;

void main() {
    vec4 t0 = texture(tex0, vUV0);
    vec4 t1 = texture(tex1, vUV1);
    vec4 r0 = vColor * t0;
    r0 *= constColor;
    r0.rgb *= t1.a * 2.0;
    r0 += vSpecColor;

    if (alphaRef.z > 0.5) {
        float cov = clamp((r0.a - alphaRef.x) / max(fwidth(r0.a), 1e-4) + 0.5, 0.0, 1.0);
        if (cov <= 0.0) discard;
        r0.a = cov;
    } else if (r0.a - alphaRef.x * alphaRef.y < 0.0) discard;

    float luminance = clamp(dot(r0.rgb, rgbEyeCoef.rgb), 0.0, 1.0);
    float nightBlend = clamp(luminance + rgbEyeCoef.a, 0.0, 1.0);
    r0.rgb = mix(vec3(luminance), r0.rgb, nightBlend);

    r0.rgb = mix(fogColor.rgb, r0.rgb, vFogTC);
    fragColor = alphaRef.w > 0.5 ? vec4(1.0, 0.0, 0.0, 1.0) : r0;
}
)";

// PSGrass — grass blending with alpha from coefficients.
const char* const kPSGrass = "#version 450\n" VK_PS_CONSTANTS_BLOCK R"(
layout(set = 0, binding = 2) uniform sampler2D tex0;
layout(set = 0, binding = 3) uniform sampler2D tex1;

layout(location = 0) in vec4 vColor;
layout(location = 1) in vec4 vSpecColor;
layout(location = 2) in vec2 vUV0;
layout(location = 3) in vec2 vUV1;
layout(location = 4) in float vFogTC;

layout(location = 0) out vec4 fragColor;

void main() {
    vec4 t0 = texture(tex0, vUV0);
    vec4 t1 = texture(tex1, vUV1);

    if (vFogTC < 0.0) discard;

    vec4 r0;
    r0.rgb = vColor.rgb * t0.rgb;
    r0.a = clamp((grassCoef1.a * 2.0 - 1.0) + t1.a, 0.0, 1.0);
    r0.rgb = clamp(r0.rgb * t1.rgb * 2.0, 0.0, 1.0);
    r0.a = clamp(grassCoef2.a * r0.a * 2.0, 0.0, 1.0);

    if (alphaRef.z > 0.5) {
        float cov = clamp((r0.a - alphaRef.x) / max(fwidth(r0.a), 1e-4) + 0.5, 0.0, 1.0);
        if (cov <= 0.0) discard;
        r0.a = cov;
    } else if (r0.a - alphaRef.x * alphaRef.y < 0.0) discard;

    r0.rgb = mix(fogColor.rgb, r0.rgb, vFogTC);
    fragColor = alphaRef.w > 0.5 ? vec4(1.0, 0.0, 0.0, 1.0) : r0;
}
)";

// PSWater — bump-mapped water with specular from light direction.
const char* const kPSWater = "#version 450\n" VK_PS_CONSTANTS_BLOCK R"(
layout(set = 0, binding = 2) uniform sampler2D tex0;
layout(set = 0, binding = 3) uniform sampler2D tex1;

layout(location = 0) in vec4 vColor;
layout(location = 1) in vec4 vSpecColor;
layout(location = 2) in vec2 vUV0;
layout(location = 3) in vec2 vUV1;
layout(location = 4) in float vFogTC;

layout(location = 0) out vec4 fragColor;

void main() {
    vec4 t0 = texture(tex0, vUV0);
    vec4 t1 = texture(tex1, vUV1);
    vec3 bumpNormal = -(t1.xyz * 2.0 - 1.0);
    float spec = clamp(dot(lightDir.xyz, bumpNormal), 0.0, 1.0);
    vec4 r0 = vColor * t0;
    r0.rgb += spec;
    r0.rgb = mix(fogColor.rgb, r0.rgb, vFogTC);
    fragColor = alphaRef.w > 0.5 ? vec4(1.0, 0.0, 0.0, 1.0) : r0;
}
)";

// PSShadow — alpha-cutout discard, constant black + vColor.a.
const char* const kPSShadow = "#version 450\n" VK_PS_CONSTANTS_BLOCK R"(
layout(set = 0, binding = 2) uniform sampler2D tex0;

layout(location = 0) in vec4 vColor;
layout(location = 1) in vec4 vSpecColor;
layout(location = 2) in vec2 vUV0;
layout(location = 3) in vec2 vUV1;
layout(location = 4) in float vFogTC;

layout(location = 0) out vec4 fragColor;

void main() {
    vec4 t0 = texture(tex0, vUV0);
    if (t0.a * vColor.a - alphaRef.x * alphaRef.y < 0.0) discard;
    fragColor = vec4(0.0, 0.0, 0.0, vColor.a);
}
)";

} // namespace

namespace Poseidon
{

bool EngineVulkan::InitShaderModules()
{
    auto build = [&](vk::ShaderStageFlagBits stage, const char* src, const char* name, vk::ShaderModule& out) {
        const std::vector<uint32_t> spirv = CompileGlslToSpirv(stage, src, name);
        if (spirv.empty())
            return false;
        out = _vk.device.createShaderModule({{}, spirv.size() * sizeof(uint32_t), spirv.data()});
        return true;
    };
    return build(vk::ShaderStageFlagBits::eVertex, kVSScreen, "vsScreen", _vsScreenModule) &&
           build(vk::ShaderStageFlagBits::eVertex, kVSTransform, "vsTransform", _vsTransformModule) &&
           build(vk::ShaderStageFlagBits::eVertex, kVSShadow, "vsShadow", _vsShadowModule) &&
           build(vk::ShaderStageFlagBits::eFragment, kPSNormal, "psNormal", _psNormalModule) &&
           build(vk::ShaderStageFlagBits::eFragment, kPSDetail, "psDetail", _psDetailModule) &&
           build(vk::ShaderStageFlagBits::eFragment, kPSGrass, "psGrass", _psGrassModule) &&
           build(vk::ShaderStageFlagBits::eFragment, kPSWater, "psWater", _psWaterModule) &&
           build(vk::ShaderStageFlagBits::eFragment, kPSFlat, "psFlat", _psFlatModule) &&
           build(vk::ShaderStageFlagBits::eFragment, kPSShadow, "psShadow", _psShadowModule);
}

void EngineVulkan::DestroyShaderModules()
{
    for (vk::ShaderModule* m : {&_vsScreenModule, &_vsTransformModule, &_vsShadowModule, &_psNormalModule,
                                &_psDetailModule, &_psGrassModule, &_psWaterModule, &_psFlatModule, &_psShadowModule})
    {
        if (*m)
            _vk.device.destroyShaderModule(*m);
        *m = nullptr;
    }
}

void EngineVulkan::ResetPSConstantDefaults()
{
    // Defaults of GL33's PSConstants struct: fog black, alpha test off,
    // IsColored tint white, night blend disabled (rgbEyeCoef.a = 1).
    static const float fog[4] = {0, 0, 0, 1};
    static const float off[4] = {0, 0, 0, 0};
    static const float white[4] = {1, 1, 1, 1};
    static const float eye[4] = {0, 0, 0, 1};
    UploadPSConstant(slots::PSFogColor, fog);
    UploadPSConstant(slots::PSAlphaRef, off);
    UploadPSConstant(slots::PSConstColor, white);
    UploadPSConstant(slots::PSRgbEyeCoef, eye);
}

void EngineVulkan::UploadPSConstant(int slot, const float* vec4)
{
    if (memcmp(_psConst + slot * 4, vec4, 16) == 0)
        return;
    memcpy(_psConst + slot * 4, vec4, 16);
    _constDirty = true;
}

void EngineVulkan::UploadVSScreenConstants()
{
    const float vpScale[4] = {2.0f / _w, 2.0f / _h, 0, 0};
    if (memcmp(_vsConst + slots::VSVpScale * 4, vpScale, 16) == 0)
        return;
    memcpy(_vsConst + slots::VSVpScale * 4, vpScale, 16);
    _constDirty = true;
}

void EngineVulkan::SetShaderFogEnabled(bool enabled)
{
    const float v = enabled ? 1.0f : 0.0f;
    if (_vsConst[slots::VSFogParam * 4 + 2] == v)
        return;
    _vsConst[slots::VSFogParam * 4 + 2] = v;
    _constDirty = true;
}

void EngineVulkan::SetAlphaTest(bool enable, unsigned ref)
{
    float alphaRef[4] = {static_cast<float>(ref) / 255.0f, enable ? 1.0f : 0.0f, 0.0f, 0.0f};
    UploadPSConstant(slots::PSAlphaRef, alphaRef);
}

void EngineVulkan::FogColorChanged(const Color& fog)
{
    const float fogColor[4] = {fog.R(), fog.G(), fog.B(), 1.0f};
    UploadPSConstant(slots::PSFogColor, fogColor);
}

} // namespace Poseidon
