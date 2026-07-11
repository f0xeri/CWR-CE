// Path-B (HW T&L) pass + constant management — structural mirror of
// EngineGL33_Mesh.cpp / the TL parts of EngineGL33_Shaders.cpp and
// EngineGL33_Material.cpp. All constant writers fill the CPU shadow blocks
// (_vsConst/_psConst); every draw snapshots them into the UBO ring, so there
// is no GL-style "flush to persistent UBO" step.
#include "EngineVulkan.hpp"
#include "TextureVulkan.hpp"

#include <Poseidon/Core/Global.hpp>
#include <Poseidon/Foundation/Logging/Logging.hpp>
#include <Poseidon/Graphics/Core/MatrixConversion.hpp>
#include <Poseidon/Graphics/Rendering/Lighting/Lights.hpp>
#include <Poseidon/World/Scene/Camera/Camera.hpp>
#include <Poseidon/World/Scene/Scene.hpp>

namespace
{

// VS UBO slot map — byte-identical mirror of PoseidonGL33's VSConst.
namespace VSC
{
enum : int
{
    SlotProj = 0,
    SlotView = 4,
    SlotWorld = 8,
    SlotSunDir = 12,
    SlotAmbient = 13,
    SlotDiffuse = 14,
    SlotEmissive = 15,
    SlotFogParam = 16,
    SlotCamPos = 17,
    SlotSpecular = 18,
    SlotSpecEn = 19,
    SlotSunEn = 20,
    SlotTexMat0 = 24,
    SlotTexMat1 = 28,
    SlotTexCtrl = 32,
    SlotLightCount = 33,
    SlotLightPos = 34,
    SlotLightDiffuse = 42,
    SlotLightAmbient = 50,
    SlotLightDir = 58,
};
constexpr int MaxLocalLights = 8;
} // namespace VSC

namespace PSC
{
enum : int
{
    SlotFogColor = 0,
    SlotConstColor = 3,
    SlotGrassCoef1 = 5,
    SlotGrassCoef2 = 6,
    SlotRgbEyeCoef = 7,
};
} // namespace PSC

// Order-sensitive signature of a draw's light list (GL33 mirror): lights are
// static within a frame and the cache resets each pass, so pointer identity
// is sufficient.
uint64_t LightsSignature(const LightList& lights)
{
    uint64_t sig = static_cast<uint64_t>(lights.Size());
    for (int i = 0; i < lights.Size(); i++)
        sig = sig * 1099511628211ull ^ reinterpret_cast<uintptr_t>(static_cast<const Poseidon::Light*>(lights[i]));
    return sig;
}

} // namespace

namespace Poseidon
{

// ── Frame-level constants ───────────────────────────────────────────────────

FrameState EngineVulkan::BuildFrameState(Camera* camera, LightSun* sun, int bias, const Color& fogColor,
                                         bool sunEnabled)
{
    FrameState frame = {};

    ConvertMatrix(frame.view, camera->InverseScaled());
    frame.view._41 = 0;
    frame.view._42 = 0;
    frame.view._43 = 0;

    const int projBias = CanZBias() ? 0 : bias;
    ConvertProjectionMatrix(frame.projection, camera->ProjectionNormal(), projBias);

    Vector3 pos = camera->Position();
    frame.cameraPos[0] = static_cast<float>(pos.X());
    frame.cameraPos[1] = static_cast<float>(pos.Y());
    frame.cameraPos[2] = static_cast<float>(pos.Z());

    frame.viewport[0] = 0;
    frame.viewport[1] = 0;
    frame.viewport[2] = static_cast<float>(_w);
    frame.viewport[3] = static_cast<float>(_h);

    float wFogStart = camera->ClipNear();
    float wFogEnd = camera->ClipFar();
    if (GScene)
    {
        wFogStart = GScene->GetFogMinRange();
        wFogEnd = GScene->GetFogMaxRange();
    }
    const float fogInvRange = (wFogEnd > wFogStart) ? 1.0f / (wFogEnd - wFogStart) : 0.0f;
    frame.fogParams[0] = wFogStart;
    frame.fogParams[1] = fogInvRange;
    frame.fogParams[2] = 1.0f; // enabled
    frame.fogParams[3] = 0;

    frame.fogColor[0] = fogColor.R();
    frame.fogColor[1] = fogColor.G();
    frame.fogColor[2] = fogColor.B();
    frame.fogColor[3] = 1.0f;

    Vector3 dir = sun->Direction();
    frame.sunDir[0] = dir.X();
    frame.sunDir[1] = dir.Y();
    frame.sunDir[2] = dir.Z();
    frame.sunDir[3] = 0;
    frame.sunEnabled = sunEnabled;

    return frame;
}

void EngineVulkan::UploadFrameConstants(const FrameState& frame)
{
    memcpy(_vsConst + VSC::SlotProj * 4, reinterpret_cast<const float*>(&frame.projection), 4 * 16);
    memcpy(_vsConst + VSC::SlotView * 4, reinterpret_cast<const float*>(&frame.view), 4 * 16);
    memcpy(_vsConst + VSC::SlotSunDir * 4, frame.sunDir, 16);

    const float sunEn[4] = {frame.sunEnabled ? 1.0f : 0.0f, 0, 0, 0};
    memcpy(_vsConst + VSC::SlotSunEn * 4, sunEn, 16);

    memcpy(_vsConst + VSC::SlotFogParam * 4, frame.fogParams, 16);

    // Camera-relative rendering: the world matrices already subtract the
    // camera position, so the shader-side camera sits at the origin.
    const float camPos[4] = {0, 0, 0, 0};
    memcpy(_vsConst + VSC::SlotCamPos * 4, camPos, 16);

    const float texCtrl[4] = {0, 0, 0, 0};
    memcpy(_vsConst + VSC::SlotTexCtrl * 4, texCtrl, 16);
    _texGenMode = render::TexGenMode::None; // keep the texgen skip-cache honest

    const float fogColor[4] = {frame.fogColor[0], frame.fogColor[1], frame.fogColor[2], 1.0f};
    UploadPSConstant(PSC::SlotFogColor, fogColor);
    _constDirty = true;
}

void EngineVulkan::UploadVSProjection(const FrameState& frame)
{
    memcpy(_vsConst + VSC::SlotProj * 4, reinterpret_cast<const float*>(&frame.projection), 64);
    _constDirty = true;
}

// ── Pass management ─────────────────────────────────────────────────────────

void EngineVulkan::BeginPass(PassId passId)
{
    if (IsIn3DPass())
    {
        _activePassId = passId;
        return;
    }
    FlushAndFreeAllQueues(_queueNo);
    _activePassId = passId;

    if (GScene)
    {
        _frameState = BuildFrameState(GScene->GetCamera(), GScene->MainLight(), _bias, _fogColor, _sunEnabled);
        _currentDrawItem = DrawItem{};
        UploadFrameConstants(_frameState);
        // Feed the (previous frame's) cascade shadow constants to the lit
        // shaders — no-op until a depth pass has run with shadow maps enabled
        // (mirrors GL33's Begin3DPass timing).
        UpdateShadowMapLitState();
        InvalidateMaterialCache();
    }
}

void EngineVulkan::UpdateProjection()
{
    if (!IsIn3DPass())
        return;
    // Pending draws must commit with the old projection before the change.
    FlushAndFreeAllQueues(_queueNo, true);
    Camera* camera = GScene->GetCamera();
    const int projBias = CanZBias() ? 0 : _bias;
    ConvertProjectionMatrix(_frameState.projection, camera->ProjectionNormal(), projBias);
    UploadVSProjection(_frameState);
}

// ── Material + lights ───────────────────────────────────────────────────────

void EngineVulkan::InvalidateMaterialCache()
{
    _materialSetSpec = -1;
    _materialSet.diffuse = Color(-1, -1, -1, -1);
    _materialSet.ambient = Color(-1, -1, -1, -1);
    _materialSet.forcedDiffuse = Color(-1, -1, -1, -1);
    _materialSet.emmisive = Color(-1, -1, -1, -1);
    _materialSet.specFlags = 0;
    _materialSetLightsSig = 0;
}

void EngineVulkan::UploadVSMaterialConstants(const TLMaterial& mat, bool /*sunEnabled*/)
{
    LightSun* sun = GScene->MainLight();

    Color dif = sun->Diffuse() * mat.diffuse;
    Color amb = sun->Ambient() * mat.ambient + sun->Diffuse() * mat.forcedDiffuse;

    const float ambient[4] = {amb.R(), amb.G(), amb.B(), amb.A()};
    const float diffuse[4] = {dif.R(), dif.G(), dif.B(), dif.A()};
    const float emissive[4] = {mat.emmisive.R(), mat.emmisive.G(), mat.emmisive.B(), mat.emmisive.A()};

    memcpy(_vsConst + VSC::SlotAmbient * 4, ambient, 16);
    memcpy(_vsConst + VSC::SlotDiffuse * 4, diffuse, 16);
    memcpy(_vsConst + VSC::SlotEmissive * 4, emissive, 16);

    Color specCol = sun->Diffuse() * mat.specular;
    const float spec[4] = {specCol.R(), specCol.G(), specCol.B(), static_cast<float>(mat.specularPower)};
    const float specEn[4] = {mat.specularPower > 0 ? 1.0f : 0.0f, 0, 0, 0};

    memcpy(_vsConst + VSC::SlotSpecular * 4, spec, 16);
    memcpy(_vsConst + VSC::SlotSpecEn * 4, specEn, 16);
    _constDirty = true;
}

void EngineVulkan::UploadVSLights(const LightList& lights, const TLMaterial& mat, float nightEffect)
{
    const float* camPos = _frameState.cameraPos;
    int n = 0;
    if (nightEffect > 0.0f)
    {
        Color matDif = mat.diffuse * nightEffect;
        Color matAmb = mat.ambient * nightEffect;
        for (int i = 0; i < lights.Size() && n < VSC::MaxLocalLights; i++)
        {
            Light* light = lights[i];
            if (!light)
                continue;
            LightDescription desc;
            light->GetDescription(desc);
            const bool isSpot = desc.type == LTSpotLight;
            if (desc.type != LTPoint && !isSpot)
                continue; // point + spot only; directional (sun) handled separately

            float* p = _vsConst + (VSC::SlotLightPos + n) * 4;
            p[0] = desc.pos.X() - camPos[0];
            p[1] = desc.pos.Y() - camPos[1];
            p[2] = desc.pos.Z() - camPos[2];
            p[3] = desc.startAtten;

            float* dir = _vsConst + (VSC::SlotLightDir + n) * 4;
            Vector3 beam = desc.dir;
            beam.Normalize();
            dir[0] = beam.X();
            dir[1] = beam.Y();
            dir[2] = beam.Z();
            dir[3] = isSpot ? 1.0f : 0.0f;

            Color dif = desc.diffuse * matDif;
            float* df = _vsConst + (VSC::SlotLightDiffuse + n) * 4;
            df[0] = dif.R();
            df[1] = dif.G();
            df[2] = dif.B();
            df[3] = 0.0f;

            Color amb = desc.ambient * matAmb;
            float* am = _vsConst + (VSC::SlotLightAmbient + n) * 4;
            am[0] = amb.R();
            am[1] = amb.G();
            am[2] = amb.B();
            am[3] = 0.0f;

            n++;
        }
    }
    _vsConst[VSC::SlotLightCount * 4] = static_cast<float>(n);
    _constDirty = true;
}

void EngineVulkan::DoSetMaterial(const TLMaterial& mat, const LightList& lights, const render::LegacySpec& spec)
{
    _materialSet = mat;
    _materialSetSpec = static_cast<int>(static_cast<std::uint32_t>(spec.material & render::Material::DisableSun));
    _materialSetLightsSig = LightsSignature(lights);

    UploadVSMaterialConstants(mat, _sunEnabled);

    // Local lights illuminate geometry only at night; DisableSun materials
    // always receive them (legacy SetupLights forced full night for those).
    float night = GScene->MainLight()->NightEffect();
    if (static_cast<std::uint32_t>(spec.material & render::Material::DisableSun) != 0)
        night = 1.0f;
    UploadVSLights(lights, mat, night);
}

void EngineVulkan::SetMaterial(const TLMaterial& mat, const LightList& lights, const render::LegacySpec& spec)
{
    const int narrowedKey = static_cast<int>(static_cast<std::uint32_t>(spec.material & render::Material::DisableSun));
    if (mat == _materialSet && _materialSetSpec == narrowedKey && LightsSignature(lights) == _materialSetLightsSig)
        return;
    DoSetMaterial(mat, lights, spec);
}

void EngineVulkan::EnableSunLight(bool enable)
{
    if (_sunEnabled == enable)
        return;
    _sunEnabled = enable;
    _frameState.sunEnabled = enable;
    if (!IsIn3DPass())
        return;
    const float sunEn[4] = {enable ? 1.0f : 0.0f, 0, 0, 0};
    memcpy(_vsConst + VSC::SlotSunEn * 4, sunEn, 16);
    _constDirty = true;
    InvalidateMaterialCache();
}

// ── TexGen (detail / grass / water UV matrices) ─────────────────────────────

void EngineVulkan::UploadVSTexGenConstants(render::TexGenMode mode)
{
    // Water re-uploads every draw (time-animated UV matrices); the other
    // modes write fixed data, so a repeat of the same mode is a no-op.
    if (mode == _texGenMode && mode != render::TexGenMode::Water)
        return;
    _texGenMode = mode;
    _constDirty = true;

    static const float identity[16] = {
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
    };
    static const float matTrans32[16] = {
        32, 0, 0, 0, 0, 32, 0, 0, 0, 0, 32, 0, 0, 0, 0, 1,
    };
    static const float matTrans64[16] = {
        64, 0, 0, 0, 0, 64, 0, 0, 0, 0, 64, 0, 0, 0, 0, 1,
    };

    if (mode == render::TexGenMode::Fixed || mode == render::TexGenMode::None)
    {
        const float texCtrl[4] = {0, 0, 0, 0};
        memcpy(_vsConst + VSC::SlotTexCtrl * 4, texCtrl, 16);
    }
    else if (mode == render::TexGenMode::Detail || mode == render::TexGenMode::Grass)
    {
        const float texCtrl[4] = {0, 1, 0, 0};
        memcpy(_vsConst + VSC::SlotTexCtrl * 4, texCtrl, 16);
        memcpy(_vsConst + VSC::SlotTexMat1 * 4, matTrans32, 4 * 16);
    }
    else if (mode == render::TexGenMode::Water)
    {
        float move[16];
        memcpy(move, identity, sizeof(move));
        float zoomAndMove[16];
        memcpy(zoomAndMove, matTrans64, sizeof(zoomAndMove));

        const float mw1 = sin(Glob.time.toFloat() * 0.04f);
        const float mw2 = fastFmod(Glob.time.toFloat() * 0.3f + sin(Glob.time.toFloat() * 0.5f) * 0.5f, 2.0f);

        move[8] = mw1 * 0.5f;
        move[9] = mw1;
        zoomAndMove[8] = mw2 * 0.5f;
        zoomAndMove[9] = mw2;

        const float texCtrl[4] = {1, 1, 0, 0};
        memcpy(_vsConst + VSC::SlotTexCtrl * 4, texCtrl, 16);
        memcpy(_vsConst + VSC::SlotTexMat0 * 4, move, 4 * 16);
        memcpy(_vsConst + VSC::SlotTexMat1 * 4, zoomAndMove, 4 * 16);
    }
}

// ── Instanced runs ──────────────────────────────────────────────────────────

bool EngineVulkan::InstancedRunAdd(const Matrix4& modelToWorld)
{
    if (_instPending >= kMaxInstances)
        return false;
    GfxMatrix& m = _instArray[_instPending];
    ConvertMatrix(m, modelToWorld);
    m._41 -= _frameState.cameraPos[0];
    m._42 -= _frameState.cameraPos[1];
    m._43 -= _frameState.cameraPos[2];
    ++_instPending;
    return true;
}

void EngineVulkan::BeginInstancedRunUpload()
{
    _instImpure = false;
    if (_instPending <= 0 || !_frameOpen)
    {
        _instCount = 0;
        return;
    }

    // The run's matrices live in a UBO-ring slice selected by binding 4's
    // dynamic offset. Always reserve the full shader-declared block (16KB)
    // so offset+range stays inside the ring for the bind-time VUID.
    VulkanBuffer& ring = _uboBuffer[_frameIndex];
    vk::DeviceSize offset = 0;
    void* dst = ring.Allocate(kWorldInstancesBytes, _uboAlign, offset);
    if (!dst)
    {
        // Ring exhausted: draw the head scalar and let the scene redraw the
        // rest (EndInstancedRun reports the run impure).
        _instCount = 1;
        _instImpure = true;
        return;
    }
    memcpy(dst, _instArray, (size_t)_instPending * 64);
    _instOffset = (uint32_t)offset;
    _instCount = _instPending;
}

// ── TL entry points ─────────────────────────────────────────────────────────

void EngineVulkan::PrepareMeshTL(const LightList& /*lights*/, const Matrix4& modelToWorld,
                                 const render::LegacySpec& spec)
{
    FlushAndFreeAllQueues(_queueNo, true);
    BeginPass(SpecToPassId(spec));
    PrepareMeshTLImpl(_frameState, modelToWorld, spec);
}

void EngineVulkan::PrepareMeshTLImpl(const FrameState& frame, const Matrix4& modelToWorld,
                                     const render::LegacySpec& spec)
{
    EnableSunLight(!render::Has(spec.material, render::Material::DisableSun));

    GfxMatrix worldMatrix;
    ConvertMatrix(worldMatrix, modelToWorld);
    // Camera-relative rendering
    worldMatrix._41 -= frame.cameraPos[0];
    worldMatrix._42 -= frame.cameraPos[1];
    worldMatrix._43 -= frame.cameraPos[2];

    _currentDrawItem = DrawItem{};
    _currentDrawItem.worldMatrix = worldMatrix;
    _currentDrawItem.specFlags = spec;
    _currentDrawItem.bias = _bias;
    // The world matrix reaches the shader as a push constant in DrawSectionTL.

    // IsColored objects carry their opacity + fade in the scene constant
    // colour (TransLight.cpp mirror).
    float constColor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    if (GScene && render::Has(spec.routing, render::Routing::IsColored))
    {
        ColorVal cc = GScene->GetConstantColor();
        constColor[0] = cc.R();
        constColor[1] = cc.G();
        constColor[2] = cc.B();
        constColor[3] = cc.A();
    }
    UploadPSConstant(PSC::SlotConstColor, constColor);
}

void EngineVulkan::BeginMeshTL(const Shape& sMesh, int /*spec*/, bool dynamic)
{
    if (VertexBuffer* buf = sMesh.GetVertexBuffer())
        buf->Update(sMesh, dynamic);
}

void EngineVulkan::EndMeshTL(const Shape& /*sMesh*/) {}

void EngineVulkan::PrepareTriangleTL(const MipInfo& mip, const render::LegacySpec& spec)
{
    ApplyPassState(mip._texture, mip._level, spec, SpecToPassId(spec), PipelineVertexInput::Mesh);
}

// ── Grass params + night eye (PS constants) ─────────────────────────────────

void EngineVulkan::SetGrassParams(float a1, float a2, float a3, float a4)
{
    _grassParam[0] = a1;
    _grassParam[1] = a2;
    _grassParam[2] = a3;
    _grassParam[3] = a4;
}

void EngineVulkan::DoSetGrassParamsPS()
{
    const float grassCoef1[4] = {0, 0, 0, _grassParam[0]};
    const float grassCoef2[4] = {0, 0, 0, _grassParam[1]};
    UploadPSConstant(PSC::SlotGrassCoef1, grassCoef1);
    UploadPSConstant(PSC::SlotGrassCoef2, grassCoef2);
}

void EngineVulkan::EnableNightEye(float night)
{
    if (_nightVision)
        night = 0;
    if (fabs(_nightEye - night) < 0.01f)
        return;
    FlushQueues();
    _nightEye = night;

    float rgbEyeCoef[4];
    if (_nightEye > 0.01f)
    {
        rgbEyeCoef[0] = 0.2f;
        rgbEyeCoef[1] = 0.9f;
        rgbEyeCoef[2] = 0.4f;
        rgbEyeCoef[3] = 1 - _nightEye;
    }
    else
    {
        rgbEyeCoef[0] = 0.0f;
        rgbEyeCoef[1] = 0.0f;
        rgbEyeCoef[2] = 0.0f;
        rgbEyeCoef[3] = 1.0f;
    }
    UploadPSConstant(PSC::SlotRgbEyeCoef, rgbEyeCoef);
}

} // namespace Poseidon
