#pragma once

#include <Poseidon/Graphics/Core/Engine.hpp>
#include <Poseidon/Graphics/Core/TLVertex.hpp>
#include <Poseidon/Graphics/Core/RenderState.hpp>
#include <PoseidonGL33/SDLEventWindow.hpp>
#include "VulkanContext.hpp"
#include "VulkanSwapchain.hpp"
#include "Utils/VulkanBuffer.hpp"

#include <unordered_map>
#include <vector>

struct SDL_Window;
class Camera; // global-namespace legacy scene camera

namespace Poseidon
{
class TextBankVulkan;
class TextureVulkan;
class VertexBufferVulkan;
class LightSun;

// Path-B mesh vertex — mirror of GL33's SVertex (pos, negated normal, uv);
// consumed by VSTransform / VSShadow.
struct SVertexVulkan
{
    Vector3P pos;
    Vector3P norm;
    UVPair t0;
};

// Path-A ("vertex soup") triangle queue — structural mirror of GL33's
// TriQueue/QueueGL33 (EngineGL33.hpp) with the texture held as the abstract
// core type. Same allocation policy, same flush cadence; only the flush
// target differs (command buffer instead of gl* calls).
struct TriQueueVulkan
{
    StaticArray<WORD> _triangleQueue;

    Texture* _texture = nullptr;
    int _level = 0;
    int _special = 0;
    PassId _passId = PassId::Opaque;
    int _lastUsed = 0;
};

struct QueueVulkan
{
    static constexpr int MaxTriQueues = 32;
    static constexpr int TriQueueSize = 2048;
    // GL33 window sizes kept 1:1 so the flush cadence (and thus draw order)
    // matches the reference backend.
    static constexpr int MeshBufferLength = 32 * 1024;
    static constexpr int IndexBufferLength = 4 * 1024;

    int _vertexBufferUsed = 0; // vertices used inside the current window
    int _indexBufferUsed = 0;  // indices used inside the current window

    int _meshBase = 0, _meshSize = 0;

    TriQueueVulkan _tri[MaxTriQueues];
    bool _triUsed[MaxTriQueues] = {};
    int _actTri = -1;

    int _usedCounter = 0;

    int Allocate(Texture* tex, int level, int spec, int minI, int maxI, int tip);
    void Free(int i);
};

class EngineVulkan : public Engine
{
    friend class VertexBufferVulkan; // uses the upload ticket + dynamic mesh ring

  private:
    // TODO(vk-phase1): TextBankVulkan; TextBankDummy for now — the producer
    // layer dereferences TextBank() unconditionally.
    AbstractTextBank* _bank;

  public:
    EngineVulkan(int width, int height, bool windowed, int bpp);
    ~EngineVulkan() override;

    bool IsUsable() const { return _sdlWindow != nullptr && _vk.IsValid(); }

    RString GetDebugName() const override;
    RString GetRendererName() const override;
    void InitDraw(bool clear = false, PackedColor color = PackedColor(0)) override;
    bool InitDrawDone() override;
    bool IsAbleToDraw() override { return IsUsable(); }
    void FinishDraw() override;
    void NextFrame() override;
    void OnWindowResized(int w, int h) override;
    void Pause() override;
    void Restore() override;
    void DrawPicture555(unsigned short*);
    void FogColorChanged(const Color&) override;
    void LightChanged(const Color&, const Color&);
    void NightEffectChanged(float);

    bool SwitchRes(int w, int h, int bpp) override;
    bool SwitchRefreshRate(int refresh) override;
    bool SetWindowMode(WindowMode mode) override;
    bool SetSwapInterval(int interval) override;
    int GetSwapInterval() const override { return _swapInterval; }
    bool GetDesktopDisplayMode(int& w, int& h, int& refresh) const override;
    bool GetRequestedFullscreenMode(int& w, int& h, int& refresh) const override;

    void HandleEvents() override;
    bool IsOpen() const override;
    void SetMouseGrab(bool grab) override;

    void ListResolutions(FindArray<ResolutionInfo>& ret) override;
    void ListRefreshRates(FindArray<int>& ret) override;

    bool CanZBias() const override;
    bool ZBiasExclusion() const override;

    int PixelSize() const override;
    int RefreshRate() const override;
    bool CanBeWindowed() const override;
    bool IsWindowed() const override;
    bool IsResizable() const override;

    void BeginMesh(TLVertexTable&, const render::LegacySpec&) override;
    void EndMesh(TLVertexTable&) override;
    AbstractTextBank* TextBank() override;

    // ── Path B: HW T&L (EngineVulkan_Mesh.cpp / EngineVulkan_VertexBuffer.cpp) ──
    bool GetTL() const override { return true; }
    bool GetTLOnSurface() const override { return true; }
    VertexBuffer* CreateVertexBuffer(const Shape& src, VBType type) override;
    void SetMaterial(const TLMaterial& mat, const LightList& lights, const render::LegacySpec& spec) override;
    void EnableSunLight(bool enable) override;
    void UpdateProjection() override;
    void PrepareMeshTL(const LightList& lights, const Matrix4& modelToWorld, const render::LegacySpec& spec) override;
    void BeginMeshTL(const Shape& sMesh, int spec, bool dynamic = false) override;
    void EndMeshTL(const Shape& sMesh) override;
    void DrawSectionTL(const Shape& sMesh, int beg, int end) override;
    void PrepareTriangleTL(const MipInfo& mip, const render::LegacySpec& spec) override;
    void SetGrassParams(float a1, float a2, float a3 = 0, float a4 = 0) override;
    void EnableNightEye(float night) override;

    void ResetForRemount() override;

    float ZShadowEpsilon() const override;
    float ZRoadEpsilon() const override;
    float ObjMipmapCoef() const override;
    float LandMipmapCoef() const;
    bool ShadowsFirst() const;
    bool SortByShape() const;

    int GetBias() override;
    void SetBias(int) override;
    void GetZCoefs(float& zAdd, float& zMult) override;

    int Width() const override;
    int Height() const override;
    int Width2D() const;
    int Height2D() const;
    int AFrameTime() const override;

    void PrepareTriangle(const PacLevelMem*, int);
    void DrawPolygon(const VertexIndex* i, int n) override;
    void DrawSection(const FaceArray&, Offset b, Offset e) override;

    void DrawDecal(Vector3Par pos, float rhw, float sizeX, float sizeY, PackedColor col, const MipInfo& mip,
                   int specFlags) override;

    using Engine::Draw2D;
    void Draw2D(const PacLevelMem*, PackedColor, float, float, float, float, float, float, float, float);

    void Clear(bool, bool, PackedColor) override;
    void SetGamma(float) override;
    float GetGamma() const override;
    void PrepareTriangle(const MipInfo&, int) override;
    void DrawPolygon(TLVertexTable&, const short*, int);
    void Draw2D(const Draw2DPars&, const Rect2DAbs&, const Rect2DAbs&) override;
    void DrawLine(int beg, int end) override;
    void DrawLine(const Line2DAbs&, PackedColor, PackedColor, const Rect2DAbs&) override;
    void PrepareMesh(const render::LegacySpec& spec) override;
    void TextureDestroyed(Texture*) override;
    float GetZCoef() const;

    void DrawPoly(const MipInfo& mip, const Vertex2DPixel* vertices, int n, const Rect2DPixel& clipRect,
                  int specFlags) override;

    void DrawPoly(const MipInfo& mip, const Vertex2DAbs* vertices, int n, const Rect2DAbs& clipRect,
                  int specFlags) override;

    // Path-A entry points shared with the producer layer (mirrors GL33).
    void DrawPoints(const TLVertex* vs, int nVertex);
    void DrawPoints(int beg, int end) override;
    void EnableReorderQueues(bool enableReorder) override;
    void FlushQueues() override;
    void BeginShadowPass() override;
    void EndShadowPass() override;

    // ── Texture upload service (EngineVulkan_Upload.cpp) ──
    // Copies are recorded into a per-frame upload command buffer submitted
    // BEFORE the frame's draw commands (doc 4.8); outside an open frame the
    // ticket degrades to a blocking one-shot submit (loading screens).
    struct UploadTicket
    {
        vk::CommandBuffer cmd;
        bool immediate = false;
        vk::CommandPool tempPool;                // immediate path only
        std::vector<VulkanBuffer> tempBuffers;   // staging-arena overflow fallbacks
    };
    UploadTicket BeginTextureUpload();
    void EndTextureUpload(UploadTicket& ticket);
    // Linear-allocates staging bytes valid for this ticket; null on OOM.
    uint8_t* AllocStaging(UploadTicket& ticket, vk::DeviceSize size, vk::Buffer& outBuffer, vk::DeviceSize& outOffset);
    // Queues GPU objects whose last CPU reference just dropped; destroyed once
    // the frame slot's fence proves the GPU is done with them.
    void DeferDestroyImage(vk::Image image, VmaAllocation alloc, vk::ImageView view);
    void DeferDestroyBuffer(vk::Buffer buffer, VmaAllocation alloc);

    VulkanContext& Context() { return _vk; }

  protected:
    int _w = 0, _h = 0;
    bool _windowed;
    int _pixelSize;
    int _refreshRate;
    int _depthBpp;
    Poseidon::WindowMode _windowMode = Poseidon::WindowMode::Borderless;
    int _windowedRestoreW = 0;
    int _windowedRestoreH = 0;
    int _bias = 0;
    float _gamma = 1.0f;

    SDL_Window* _sdlWindow = nullptr;
    SDLEventWindow _eventWindow;
    VulkanContext _vk;
    VulkanSwapchain _swapchain;

    static constexpr int kFramesInFlight = 2;
    struct FrameResources
    {
        vk::CommandPool pool;
        vk::CommandBuffer cmd;
        vk::CommandBuffer uploadCmd; // texture copies, submitted before cmd
        vk::Fence inFlight;
        vk::Semaphore imageAvailable;
    };
    FrameResources _frames[kFramesInFlight];
    int _frameIndex = 0;
    uint32_t _imageIndex = 0;
    // Same contract as GL33's _frameOpen; the extra _frameRecorded step exists
    // because in Vulkan the closed command buffer still has to be submitted.
    bool _frameOpen = false;     // between successful InitDraw and FinishDraw
    bool _frameRecorded = false; // between FinishDraw and NextFrame
    bool _swapchainDirty = false; // rebuild before next acquire
    int _swapInterval = 1;        // GL semantics: 1 = vsync, 0 = off, -1 = adaptive
    PackedColor _clearColor{0xFF203040u};
    bool ApplyExclusiveFullscreen(int w, int h, int refresh);

    bool CreateFrameResources();
    void DestroyFrameResources();
    void RecreateSwapchain();

    // ── Texture upload plumbing (EngineVulkan_Upload.cpp) ──
    VulkanBuffer _stagingBuffer[kFramesInFlight]; // per-frame staging arena
    bool _uploadOpen = false; // current slot's uploadCmd is recording
    struct DeferredImage
    {
        vk::Image image;
        VmaAllocation alloc = nullptr;
        vk::ImageView view;
    };
    struct DeferredRawBuffer
    {
        vk::Buffer buffer;
        VmaAllocation alloc = nullptr;
    };
    std::vector<DeferredImage> _deferredImages[kFramesInFlight];
    std::vector<VulkanBuffer> _deferredBuffers[kFramesInFlight];
    std::vector<DeferredRawBuffer> _deferredRawBuffers[kFramesInFlight];
    int DeferSlot() const; // slot whose fence covers the newest GPU work
    void FlushDeferredDestroys(int slot); // slot's fence must be signaled
    void FlushAllDeferredDestroys();      // device must be idle

    // ── Path A: soup queues + screen pipeline (EngineVulkan_Queue/_2D/_Pipeline/_Shaders) ──

    enum RenderMode
    {
        RMLines,
        RMTris,
        RM2DLines,
        RM2DTris
    };
    RenderMode _renderMode = RMTris;
    void SwitchRenderMode(RenderMode mode)
    {
        if (_renderMode != mode)
            DoSwitchRenderMode(mode);
    }
    void DoSwitchRenderMode(RenderMode mode);

    PassId _activePassId = PassId::ScreenSpace;
    bool IsIn3DPass() const { return _activePassId != PassId::ScreenSpace; }
    void BeginScreenPass();

    QueueVulkan _queueNo;
    bool _enableReorder = true;
    int _prepSpec = 0;
    TLVertexTable* _mesh = nullptr;

    // Per-frame transient allocators (host-visible, persistently mapped).
    // Reset after frame fence signals. Allocations are valid only within one frame.
    VulkanBuffer _vertexBuffer[kFramesInFlight];
    VulkanBuffer _indexBuffer[kFramesInFlight];
    VulkanBuffer _uboBuffer[kFramesInFlight];
    VulkanBuffer _dynMeshBuffer[kFramesInFlight]; // path-B dynamic (animated) mesh vertices
    void* AllocDynamicMeshVertices(vk::DeviceSize bytes, vk::Buffer& outBuffer, vk::DeviceSize& outOffset);
    int _vertexWindowBase = 0; // in vertices, into the current frame's vertex ring
    int _indexWindowBase = 0;  // in indices, into the current frame's index ring
    bool _soupOverflowLogged = false;

    WORD* QueueAdd(QueueVulkan& queue, int n);
    void QueueFan(const VertexIndex* ii, int n);
    void Queue2DPoly(const TLVertex* v0, int n);
    void FlushQueue(QueueVulkan& queue, int index);
    void FlushAndFreeQueue(QueueVulkan& queue, int index);
    int AllocateQueue(QueueVulkan& queue, Texture* tex, int level, int spec);
    void FreeQueue(QueueVulkan& queue, int index);
    void FreeAllQueues(QueueVulkan& queue);
    void FlushAndFreeAllQueues(QueueVulkan& queue, bool nonEmptyOnly = false);
    void FlushAllQueues(QueueVulkan& queue, int skip = -1);
    void QueuePrepareTriangle(const MipInfo& absMip, int specFlags);
    void AddVertices(const TLVertex* v, int n);

    // ── Shaders + constants (EngineVulkan_Shaders.cpp) ──
    // CPU shadow copies of the shared std140 UBO blocks; every flush writes a
    // fresh snapshot into the UBO ring (slot layout = GL33's VSConst/PSConstants).
    static constexpr int kVSConstFloats = 280; // 70 vec4 slots
    static constexpr int kPSConstFloats = 108; // 27 vec4 slots
    float _vsConst[kVSConstFloats] = {};
    float _psConst[kPSConstFloats] = {};

    vk::ShaderModule _vsScreenModule;
    vk::ShaderModule _vsTransformModule;
    vk::ShaderModule _vsShadowModule;
    vk::ShaderModule _psNormalModule;
    vk::ShaderModule _psDetailModule;
    vk::ShaderModule _psGrassModule;
    vk::ShaderModule _psWaterModule;
    vk::ShaderModule _psFlatModule;
    vk::ShaderModule _psShadowModule;

    bool InitShaderModules();
    void DestroyShaderModules();
    void ResetPSConstantDefaults();
    void UploadVSScreenConstants();
    void SetShaderFogEnabled(bool enabled);
    void SetAlphaTest(bool enable, unsigned ref);
    void UploadPSConstant(int slot, const float* vec4);

    // ── Path B state: frame/material constants (EngineVulkan_Mesh.cpp) ──
    // GL33's VSConst slot map is mirrored in _vsConst; these writers fill the
    // CPU shadow, the per-draw descriptor write snapshots it into the ring.
    FrameState _frameState;
    bool _sunEnabled = true;
    DrawItem _currentDrawItem;
    TLMaterial _materialSet;
    int _materialSetSpec = -1;
    uint64_t _materialSetLightsSig = 0;
    float _grassParam[4] = {};
    float _nightEye = 0.0f;

    void BeginPass(PassId passId);
    FrameState BuildFrameState(Camera* camera, LightSun* sun, int bias, const Color& fogColor, bool sunEnabled);
    void UploadFrameConstants(const FrameState& frame);
    void UploadVSProjection(const FrameState& frame);
    void UploadVSMaterialConstants(const TLMaterial& mat, bool sunEnabled);
    void UploadVSLights(const LightList& lights, const TLMaterial& mat, float nightEffect);
    void UploadVSTexGenConstants(render::TexGenMode mode);
    void DoSetMaterial(const TLMaterial& mat, const LightList& lights, const render::LegacySpec& spec);
    void PrepareMeshTLImpl(const FrameState& frame, const Matrix4& modelToWorld, const render::LegacySpec& spec);
    void InvalidateMaterialCache();
    void DoSetGrassParamsPS();

    // ── Pipeline cache + descriptors (EngineVulkan_Pipeline.cpp) ──
    enum class PSSel : uint8_t
    {
        Normal,
        Detail,
        Grass,
        Water,
        Flat,
        Shadow
    };

    // Vertex-input flavor baked into the pipeline: Screen = TLVertex +
    // VSScreen, Mesh = SVertexVulkan + VSTransform (VSShadow for PSSel::Shadow).
    enum class PipelineVertexInput : uint8_t
    {
        ActivePass,
        Screen,
        Mesh
    };

    struct PipelineKey
    {
        PSSel ps = PSSel::Normal;
        uint8_t mesh = 0;      // 1 = mesh vertex input (SVertexVulkan)
        uint8_t blend = 0;     // render::BlendMode
        uint8_t depth = 0;     // render::DepthMode
        uint8_t cull = 0;      // render::CullMode
        uint8_t frontFace = 0; // render::FrontFaceMode

        bool operator==(const PipelineKey& r) const
        {
            return ps == r.ps && mesh == r.mesh && blend == r.blend && depth == r.depth && cull == r.cull &&
                   frontFace == r.frontFace;
        }
    };
    struct PipelineKeyHash
    {
        size_t operator()(const PipelineKey& k) const
        {
            return (size_t)k.ps | ((size_t)k.mesh << 4) | ((size_t)k.blend << 8) | ((size_t)k.depth << 16) |
                   ((size_t)k.cull << 24) | ((size_t)k.frontFace << 32);
        }
    };

    std::unordered_map<PipelineKey, vk::Pipeline, PipelineKeyHash> _pipelines;
    PipelineKey _lastBoundKey;
    bool _pipelineBound = false;

    vk::DescriptorSetLayout _setLayout;
    vk::PipelineLayout _pipelineLayout;
    static constexpr int kSamplerCombos = 8; // filter(2) x clampU(2) x clampV(2)
    vk::Sampler _samplers[kSamplerCombos];
    // Per-flush descriptor sets from a per-frame pool (doc 4.5): each flush
    // allocates a fresh set (2 dynamic UBOs + the flush's texture view); the
    // whole pool is reset in InitDraw once the slot's fence signals.
    vk::DescriptorPool _frameDescPool[kFramesInFlight];
    int _currentSamplerIdx = 0;
    vk::ImageView _currentTexView;  // resolved by ApplyPassState; null = white
    vk::ImageView _currentTex1View; // detail/grass/specular slot; null = white
    uint32_t _uboAlign = 256;

    // Descriptor reuse (perf): the world matrix travels as a push constant,
    // so consecutive draws whose UBO shadow + textures are unchanged rebind
    // the previous set instead of snapshotting 1.5KB and allocating a new one.
    bool _constDirty = true;                // any _vsConst/_psConst write sets this
    render::TexGenMode _texGenMode = render::TexGenMode::None;
    vk::DescriptorSet _cachedSet;           // last written set (this frame's pool)
    uint32_t _cachedDynOffsets[2] = {};
    vk::ImageView _cachedTex0;
    vk::ImageView _cachedTex1;
    int _cachedSampler = -1;

    vk::Image _whiteImage;
    VmaAllocation _whiteAlloc = nullptr;
    vk::ImageView _whiteView;

    vk::Format _depthFormat = vk::Format::eUndefined;
    vk::Image _depthImage;
    VmaAllocation _depthAlloc = nullptr;
    vk::ImageView _depthView;

    bool InitPipelineResources(); // rings, samplers, white tex, layouts, descriptor sets
    void DestroyPipelineResources();
    bool CreateDepthTarget();
    void DestroyDepthTarget();
    vk::Pipeline GetOrCreatePipeline(const PipelineKey& key);
    void ApplyPassState(Texture* tex, int level, const render::LegacySpec& spec, PassId passId,
                        PipelineVertexInput vertexInput = PipelineVertexInput::Screen);
    bool WriteConstantsAndBindDescriptors(vk::CommandBuffer cmd);
};

Engine* CreateEngineVulkan(int w, int h, bool windowed, int bpp);

} // namespace Poseidon
