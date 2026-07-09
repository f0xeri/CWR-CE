#pragma once

// Texture + bank — structural mirror of PoseidonGL33/TextureGL33.hpp with the
// GPU surface swapped for VkImage+VMA. Same demand-streaming model: headers on
// first touch, small surface + big surface, per-frame LRU lists, byte budgets.
// Uploads go through EngineVulkan's upload command buffer (executed before the
// frame's draws); destruction is deferred until the GPU is provably done.

#include <Poseidon/Graphics/Textures/TextureBank.hpp>
#include <Poseidon/Graphics/Rendering/Colors.hpp>
#include <Poseidon/Foundation/Memory/MemFreeReq.hpp>

#include "VulkanContext.hpp"

namespace Poseidon
{

class TextureVulkan;
class TextBankVulkan;
class EngineVulkan;

class HMipCacheVulkan : public CLDLink
{
  public:
    TextureVulkan* texture;

    USE_FAST_ALLOCATOR;
};

typedef CLList<HMipCacheVulkan> VulkanMipCacheRoot;

struct TextureDescVulkan
{
    int w, h;
    int nMipmaps;
    vk::Format vkFormat;
    bool compressed;
};

// PacFormat -> VkFormat (eUndefined when there is no native equivalent).
vk::Format PacToVkFormat(PacFormat format);
int MipmapSizeVulkan(PacFormat format, int w, int h);

class SurfaceInfoVulkan
{
  private:
    vk::Image _image;
    vk::ImageView _view;
    VmaAllocation _alloc = nullptr;

    static int _nextId;
    int _id;

  public:
    int _totalSize;
    int _usedSize;
    int _w, _h;
    int _nMipmaps;
    PacFormat _format;

    int SizeExpected() const { return _totalSize; }
    int SizeUsed() const { return _usedSize; }
    int GetCreationID() const { return _id; }

    static int CalculateSize(const TextureDescVulkan& desc, PacFormat format, int totalSize = -1);
    int CreateSurface(EngineVulkan* engine, const TextureDescVulkan& desc, PacFormat format, int totalSize = -1);
    // lastRef=true: hand the image to the engine's deferred-delete queue
    // (in-flight frames may still sample it); lastRef=false: forget the
    // handles (ownership moved to another SurfaceInfoVulkan).
    void Free(bool lastRef, int refValue = 0);

    bool HasImage() const { return static_cast<bool>(_image); }
    vk::Image GetImage() const { return _image; }
    vk::ImageView GetView() const { return _view; }
    void TakeOver(SurfaceInfoVulkan& from);
};

#define ASSERT_INIT_VK() PoseidonAssert(_initialized);

class TextureVulkan : public Texture
{
    typedef Texture base;

    friend class TextBankVulkan;
    friend class EngineVulkan;

  private:
    SRef<ITextureSource> _src;

    Ref<TextureVulkan> _interpolate;
    float _iFactor;

    bool _isDetail;
    bool _useDetail;
    bool _initialized;
    bool _dynamicMipmapped = false; // UpdateRGBA regenerates the mip chain when true

    int _maxSize;

    int _nMipmaps;
    PacLevelMem _mipmaps[MAX_MIPMAPS];

    SurfaceInfoVulkan _surface;      // GPU texture
    SurfaceInfoVulkan _smallSurface; // small GPU texture

    signed char _alphaClass = -1; // cached GetAlphaClass() verdict (-1 = not computed)

    signed char _largestUsed;
    signed char _smallLoaded;
    signed char _levelLoaded;
    signed char _levelNeededThisFrame;
    signed char _levelNeededLastFrame;
    signed char _inUse;

    HMipCacheVulkan* _cache;

    int LevelNeeded() const
    {
        return (_levelNeededThisFrame < _levelNeededLastFrame ? _levelNeededThisFrame : _levelNeededLastFrame);
    }

  public:
    TextureVulkan();
    ~TextureVulkan() override;

    void InitDesc(TextureDescVulkan& desc, int levelMin);

    int LoadLevels(int levelMin);
    int UploadToGPU(SurfaceInfoVulkan& surface, int levelMin);

    // The view a flush should sample this frame: big surface when resident,
    // small otherwise, null when nothing is on the GPU yet (caller falls back
    // to the engine's white texture).
    vk::ImageView GetSampledView() const
    {
        vk::ImageView view = _surface.GetView();
        return view ? view : _smallSurface.GetView();
    }

    const SurfaceInfoVulkan& GetSurface() const { return _surface.HasImage() ? _surface : _smallSurface; }

  private:
    void MemoryReleased();
    AlphaStats::Kind ScanTopMipAlphaClass();

    int TotalSize(int levelMin) const;
    void ReleaseSmall(bool store = false);
    int LoadSmall();

  public:
    void ReleaseMemory(bool store = false);
    void ReuseMemory(SurfaceInfoVulkan& surf);

    bool IsAlpha() const override
    {
        ASSERT_INIT_VK();
        return _src && _src->IsAlpha();
    }
    int AMaxSize() const override { return _maxSize; }
    void SetMaxSize(int size) override;
    void SetMultitexturing(int type) override;

    bool VerifyChecksum(const MipInfo& mip) const override;

    void SetMipmapRange(int min, int max) override;

    int Init(const char* name);

    bool InitFromRGBA(int w, int h, const void* rgba, uint32_t size, bool mipmap = false);
    void UpdateRGBA(const void* rgba, uint32_t size);
    void DoLoadHeaders();
    void PreloadHeaders();

    void LoadHeadersNV() const
    {
        if (_initialized)
            return;
        const_cast<TextureVulkan*>(this)->DoLoadHeaders();
    }

    void LoadHeaders() override;

    const PacLevelMem* Mipmap(int level) const
    {
        ASSERT_INIT_VK();
        return &_mipmaps[level];
    }

    const AbstractMipmapLevel& AMipmap(int level) const override
    {
        ASSERT_INIT_VK();
        return _mipmaps[level];
    }
    AbstractMipmapLevel& AMipmap(int level) override
    {
        ASSERT_INIT_VK();
        return _mipmaps[level];
    }

    int NMipmaps() const
    {
        ASSERT_INIT_VK();
        return _nMipmaps;
    }
    int ANMipmaps() const override
    {
        ASSERT_INIT_VK();
        return _nMipmaps;
    }
    void ASetNMipmaps(int n) override;
    int AWidth(int level = 0) const override
    {
        LoadHeadersNV();
        return _mipmaps[level]._w;
    }
    int AHeight(int level = 0) const override
    {
        LoadHeadersNV();
        return _mipmaps[level]._h;
    }
    bool IsTransparent() const override
    {
        ASSERT_INIT_VK();
        return _src && _src->IsTransparent();
    }

    AlphaStats::Kind GetAlphaClass() override;

    Color GetPixel(int level, float u, float v) const override;
    Color GetColor() override
    {
        LoadHeaders();
        return _src ? _src->GetAverageColor() : HBlack;
    }

    int Width(int level) const
    {
        ASSERT_INIT_VK();
        return _mipmaps[level]._w;
    }
    int Height(int level) const
    {
        ASSERT_INIT_VK();
        return _mipmaps[level]._h;
    }

    void CacheUse(VulkanMipCacheRoot& list);

    NoCopy(TextureVulkan);
    USE_FAST_ALLOCATOR
};

class TextBankVulkan : public AbstractTextBank
{
    typedef AbstractTextBank base;

    friend class TextureVulkan;

    int _maxTextureMemory;
    int _limitAllocatedTextures;
    int _reserveTextureMemory;

    int _maxSmallTexturePixels;

    LLinkArray<TextureVulkan> _texture;
    Ref<TextureVulkan> _detail;
    Ref<TextureVulkan> _specular;
    Ref<TextureVulkan> _grass;
    Ref<TextureVulkan> _waterBump;

    AutoArray<SurfaceInfoVulkan> _freeTextures;

    int _totalAllocated;

    EngineVulkan* _engine;

    Foundation::MemoryDomainProbe _memProbe;

    VulkanMipCacheRoot _thisFrameWholeUsed;
    VulkanMipCacheRoot _lastFrameWholeUsed;
    VulkanMipCacheRoot _thisFramePartialUsed;
    VulkanMipCacheRoot _lastFramePartialUsed;
    VulkanMipCacheRoot _previousUsed;

    int _thisFrameCopied;
    int _loadBoostFrames = 0;
    int _thisFrameAlloc;

  public:
    TextBankVulkan(EngineVulkan* engine);
    ~TextBankVulkan() override;

    int FreeTextureMemory();

    EngineVulkan* GetEngine() const { return _engine; }
    // Native sampled-image support for a Pac format on this device;
    // DoLoadHeaders falls back to PacARGB8888 when false.
    bool NativeFormatSupported(PacFormat format) const;

  private:
    void CheckTextureMemory();
    void InitDetailTextures();

  public:
    int NTextures() const override { return _texture.Size(); }
    Texture* GetTexture(int i) const override { return _texture[i]; }

    void Compact() override;

    void Preload() override;
    void FlushTextures() override;
    void ForceReloadAll() override;
    void FlushBank(QFBank* bank) override;

  protected:
    int FindFree();

    TextureVulkan* Copy(int from);
    int Find(RStringB name1, TextureVulkan* interpolate = nullptr);

    int FindSurface(int w, int h, int nMipmaps, PacFormat format, const AutoArray<SurfaceInfoVulkan>& array) const;

    int FindReleased(int w, int h, int nMipmaps, PacFormat format) const
    {
        return FindSurface(w, h, nMipmaps, format, _freeTextures);
    }
    void AddReleased(SurfaceInfoVulkan& surf);
    void UseReleased(SurfaceInfoVulkan& surf, const TextureDescVulkan& desc, PacFormat format);

    void Reuse(SurfaceInfoVulkan& surf, const TextureDescVulkan& desc, PacFormat format);
    void DeleteLastReleased();

  public:
    Ref<Texture> Load(RStringB name) override;
    Ref<Texture> LoadInterpolated(RStringB n1, RStringB n2, float factor) override;

    Texture* CreateDynamic(int w, int h, const void* rgba, uint32_t size, bool mipmap = false) override;
    void UpdateDynamic(Texture* tex, const void* rgba, uint32_t size) override;

    void StopAll();

    void StartFrame() override;
    void BoostLoadBudget(int frames) override
    {
        if (frames > _loadBoostFrames)
            _loadBoostFrames = frames;
    }
    void FinishFrame() override;

    MipInfo UseMipmap(Texture* texture, int level, int levelTop) override;
    TextureVulkan* GetDetailTexture() const { return _detail; }
    TextureVulkan* GetGrassTexture() const { return _grass; }
    TextureVulkan* GetSpecularTexture() const { return _specular; }
    TextureVulkan* GetWaterBumpMap() const { return _waterBump; }

    bool VerifyChecksums() override;
    bool ReserveMemory(VulkanMipCacheRoot& root, int limit);
    bool ReserveMemory(int size);
    bool ForcedReserveMemory(int size);
    void ReleaseAllTextures() override;

    int CreateGPUSurface(SurfaceInfoVulkan& surface, const TextureDescVulkan& desc, PacFormat format, int totalSize);

    void ReportTextures(const char* name);

    NoCopy(TextBankVulkan);
};

} // namespace Poseidon
