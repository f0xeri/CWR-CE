// Mirror of PoseidonGL33/TextureGL33_Init.cpp: header loading, format
// selection, surface creation, dynamic (raw RGBA) textures. GL's
// internalFormat/pixelFormat/pixelType triple collapses into one vk::Format;
// unsupported 16-bit packed formats fall back to PacARGB8888 (the source
// decoder converts on CPU, same mechanism GL33 uses for its fallbacks).
#include <Poseidon/Core/Application.hpp>
#include <Poseidon/Core/Config/EngineConfig.hpp>

#include <Poseidon/IO/Streams/QBStream.hpp>
#include <Poseidon/Foundation/Algorithms/Qsort.hpp>

#include "TextureVulkan.hpp"
#include "EngineVulkan.hpp"
#include <Poseidon/IO/FileServer.hpp>
#include <Poseidon/Core/Global.hpp>
#include <Poseidon/Graphics/Textures/LooseTextures.hpp>
#include <Poseidon/Graphics/Textures/PAADecoder.hpp>

#include <cstring>

namespace Poseidon
{

int MipmapSizeVulkan(PacFormat format, int w, int h)
{
    switch (format)
    {
        case PacDXT1:
            return ((w + 3) / 4) * ((h + 3) / 4) * 8;
        case PacDXT2:
        case PacDXT3:
        case PacDXT4:
        case PacDXT5:
            return ((w + 3) / 4) * ((h + 3) / 4) * 16;
        case PacARGB8888:
            return w * h * 4;
        default:
            return w * h * 2; // 16-bit formats
    }
}

vk::Format PacToVkFormat(PacFormat format)
{
    switch (format)
    {
        case PacDXT1:
            return vk::Format::eBc1RgbaUnormBlock;
        case PacDXT2:
        case PacDXT3:
            return vk::Format::eBc2UnormBlock;
        case PacDXT4:
        case PacDXT5:
            return vk::Format::eBc3UnormBlock;
        // Little-endian 0xAARRGGBB dword = B,G,R,A bytes (GL uploaded as
        // GL_BGRA + 8888_REV for the same reason).
        case PacARGB8888:
            return vk::Format::eB8G8R8A8Unorm;
        case PacARGB1555:
            return vk::Format::eA1R5G5B5UnormPack16;
        case PacRGB565:
            return vk::Format::eR5G6B5UnormPack16;
        case PacARGB4444:
            return vk::Format::eA4R4G4B4UnormPack16;
        case PacAI88:
            return vk::Format::eR8G8Unorm; // swizzled (R,R,R,G) in the view
        case PacP8:
            return vk::Format::eR8Unorm;
        default:
            return vk::Format::eUndefined;
    }
}

static PacFormat BasicFormat(const char* name)
{
    const char* ext = strrchr(name, '.');
    if (ext && !strcmpi(ext, ".paa"))
    {
        return PacARGB4444;
    }
    return PacARGB1555;
}

static PacFormat DstFormat(TextBankVulkan* bank, PacFormat srcFormat)
{
    switch (srcFormat)
    {
        case PacARGB1555:
        case PacRGB565:
        case PacARGB4444:
            return bank->NativeFormatSupported(srcFormat) ? srcFormat : PacARGB8888;
        case PacP8:
            return bank->NativeFormatSupported(PacARGB1555) ? PacARGB1555 : PacARGB8888;
        case PacAI88:
        case PacDXT1:
            // R8G8 and BC1-3 are universal on any desktop Vulkan 1.3 device.
            return srcFormat;
        default:
            return srcFormat;
    }
}

void InitVkPixelFormat(TextureDescVulkan& desc, PacFormat format)
{
    desc.vkFormat = PacToVkFormat(format);
    desc.compressed = (format == PacDXT1 || format == PacDXT2 || format == PacDXT3 || format == PacDXT4 ||
                       format == PacDXT5);
    if (desc.vkFormat == vk::Format::eUndefined)
        Foundation::ErrorMessage("Texture has bad pixel format (VK).");
    if (format == PacP8)
        Fail("Palette textures obsolete");
}

#define MIN_MIP_SIZE 4

void TextureVulkan::SetMipmapRange(int min, int max)
{
    if (min < 0)
        min = 0;
    if (max > _nMipmaps - 1)
        max = _nMipmaps - 1;
    if (min > max)
        min = max;
    _largestUsed = min;
    _nMipmaps = max + 1;
}

int TextureVulkan::Init(const char* name)
{
    SetName(name);

    _maxSize = 0x10000;

    RString resolved = Graphics::ResolveLooseTexturePath(name);
    ITextureSourceFactory* factory = SelectTextureSourceFactory(resolved);
    if (!factory || !factory->Check(resolved))
    {
        Foundation::WarningMessage("Cannot load texture %s.", static_cast<const char*>(GetName()));
        _nMipmaps = 0;
        return -1;
    }

    return 0;
}

void TextureVulkan::PreloadHeaders()
{
    RString resolved = Graphics::ResolveLooseTexturePath(Name());
    ITextureSourceFactory* factory = SelectTextureSourceFactory(resolved);
    if (!factory)
        return;
    factory->PreInit(resolved);
}

void TextureVulkan::DoLoadHeaders()
{
    PoseidonAssert(!_initialized);
    _initialized = true;

    int i = -1;

    PacFormat format = BasicFormat(GetName());
    bool isPaa = (format == PacARGB4444);

    TextBankVulkan* bank = static_cast<TextBankVulkan*>(GEngine->TextBank());

    if (_maxSize >= 0x10000)
    {
        if (!CmpStartStr(Name(), "fonts\\"))
            _maxSize = 1024;
        else if (!CmpStartStr(Name(), "merged\\"))
            _maxSize = 2048;
        else if (bank->AnimatedNumber(Name()) >= 0 && IsAlpha())
            _maxSize = ENGINE_CONFIG.maxAnimText;
        else
            _maxSize = ENGINE_CONFIG.maxObjText;
    }

    RString resolved2 = Graphics::ResolveLooseTexturePath(Name());
    ITextureSourceFactory* factory = SelectTextureSourceFactory(resolved2);
    if (!factory)
    {
        _nMipmaps = 0;
        return;
    }
    _src = factory->Create(resolved2, _mipmaps, MAX_MIPMAPS);

    if (_src)
    {
        format = _src->GetFormat();

        if (format == PacARGB4444 || format == PacAI88 || format == PacARGB8888)
            _src->ForceAlpha();

        PacFormat dFormat = DstFormat(bank, format);

        if (!_src->IsTransparent() && _src->GetFormat() == PacARGB1555)
        {
            if (bank->NativeFormatSupported(PacRGB565) && !isPaa)
                dFormat = PacRGB565;
        }

        _largestUsed = MAX_MIPMAPS;
        int nMipmaps = _src->GetMipmapCount();
        for (i = 0; i < nMipmaps; i++)
        {
            PacLevelMem& mip = _mipmaps[i];
            mip.SetDestFormat(dFormat, 8);

            if (!mip.TooLarge(_maxSize))
            {
                if (_largestUsed > i)
                    _largestUsed = i;
            }

            if (mip._w < MIN_MIP_SIZE)
                break;
            if (mip._h < MIN_MIP_SIZE)
                break;
        }

        _nMipmaps = i;
        _levelLoaded = i;
        _smallLoaded = i;
        _levelNeededThisFrame = _levelNeededLastFrame = i;

        return;
    }

    Foundation::WarningMessage("Cannot load texture %s.", static_cast<const char*>(GetName()));
    _nMipmaps = 0;
}

void TextureVulkan::LoadHeaders()
{
    if (_initialized)
        return;
    DoLoadHeaders();
}

Color TextureVulkan::GetPixel(int level, float u, float v) const
{
    LoadHeadersNV();

    QIFStream in;
    GFileServer->Open(in, Name());
    if (in.fail())
        return HWhite;

    Color icol;
    QIFStream ipol;
    if (_interpolate)
    {
        GFileServer->Open(ipol, _interpolate->Name());
        if (ipol.fail())
            return HWhite;
    }

    PacLevelMem mip = _mipmaps[level];

    AUTO_STATIC_ARRAY(char, mem, 256 * 256 * 2);
    mem.Realloc(mip._pitch * mip._h);
    mem.Resize(mip._pitch * mip._h);

    _src->GetMipmapData(mem.Data(), mip, level);

    if (_interpolate)
    {
        AUTO_STATIC_ARRAY(char, imem, 256 * 256 * 2);
        PacLevelMem& imip = _interpolate->_mipmaps[level];
        imem.Realloc(imip._pitch * imip._h);
        imem.Resize(imip._pitch * imip._h);
        _interpolate->_src->GetMipmapData(imem.Data(), imip, level);
        icol = imip.GetPixel(imem.Data(), u, v);
    }
    Color col = mip.GetPixel(mem.Data(), u, v);
    if (_interpolate)
    {
        col = col * (1 - _iFactor) + icol * _iFactor;
    }
    return col;
}

// See TextureGL33::ScanTopMipAlphaClass for the reasoning (top mip on
// purpose: smaller mips blur crisp cutout holes into false partial alpha).
AlphaStats::Kind TextureVulkan::ScanTopMipAlphaClass()
{
    LoadHeadersNV();
    if (!_src)
        return AlphaStats::Opaque;

    QIFStream in;
    GFileServer->Open(in, Name());
    const int size = in.fail() ? 0 : in.rest();
    if (size <= 0)
        return AlphaStats::Opaque;

    AUTO_STATIC_ARRAY(char, fileData, 256 * 1024);
    fileData.Realloc(size);
    fileData.Resize(size);
    in.read(fileData.Data(), size);

    const char* name = Name();
    const size_t len = name ? strlen(name) : 0;
    const bool isPaa = len >= 4 && (name[len - 1] == 'a' || name[len - 1] == 'A'); // .paa vs .pac

    const DecodedImage img = DecodePAABuffer(fileData.Data(), static_cast<size_t>(size), isPaa);
    if (!img.valid())
        return AlphaStats::Opaque;

    return ClassifyAlpha(img.rgba.data(), static_cast<size_t>(img.width) * static_cast<size_t>(img.height)).kind;
}

AlphaStats::Kind TextureVulkan::GetAlphaClass()
{
    if (_alphaClass >= 0)
        return static_cast<AlphaStats::Kind>(_alphaClass);

    LoadHeadersNV();
    AlphaStats::Kind kind = AlphaStats::Opaque;
    if (_src)
    {
        const bool hasAlpha = _src->IsAlpha();
        const bool chroma = _src->IsTransparent();
        const bool oneBit = _src->GetFormat() == PacDXT1;
        AlphaStats decoded;
        const AlphaStats* decodedPtr = nullptr;
        if (hasAlpha && !oneBit)
        {
            decoded.kind = ScanTopMipAlphaClass();
            decodedPtr = &decoded;
        }
        kind = ClassifyTextureAlpha(hasAlpha, chroma, oneBit, decodedPtr);
    }
    _alphaClass = static_cast<signed char>(kind);
    return kind;
}

DEFINE_FAST_ALLOCATOR(TextureVulkan);
DEFINE_FAST_ALLOCATOR(HMipCacheVulkan);

TextureVulkan::TextureVulkan()
    : _nMipmaps(0), _levelLoaded(63), _smallLoaded(63), _levelNeededThisFrame(0), _levelNeededLastFrame(0),
      _isDetail(false), _useDetail(false), _cache(nullptr), _inUse(0), _interpolate(nullptr), _maxSize(256),
      _initialized(false)
{
}

TextureVulkan::~TextureVulkan()
{
    ReleaseMemory(false);
    ReleaseSmall(false);
    GEngine->TextureDestroyed(this);
}

void TextureVulkan::SetMaxSize(int size)
{
    LoadHeadersNV();
    if (size >= _maxSize)
        return;
    _maxSize = size;
}

void TextureVulkan::SetMultitexturing(int type)
{
    _useDetail = (type != 0);
}

bool TextureVulkan::VerifyChecksum(const MipInfo&) const
{
    return true;
}

void TextureVulkan::ASetNMipmaps(int n)
{
    LoadHeadersNV();
    if (n > _nMipmaps)
    {
        LOG_ERROR(Graphics, "Out of range ASetNMipmaps in {}", static_cast<const char*>(GetName()));
        n = _nMipmaps;
    }
    _nMipmaps = n;
    PacLevelMem& mip = _mipmaps[n - 1];
    saturateMax(_maxSize, mip._w);
    saturateMax(_maxSize, mip._h);
    if (_largestUsed > _nMipmaps - 1)
        _largestUsed = _nMipmaps - 1;
    if (_interpolate)
        _interpolate->ASetNMipmaps(n);
}

// ── Surface ─────────────────────────────────────────────────────────────────

int SurfaceInfoVulkan::_nextId = 0;

int SurfaceInfoVulkan::CalculateSize(const TextureDescVulkan& desc, PacFormat format, int totalSize)
{
    if (totalSize >= 0)
        return totalSize;

    int size = 0;
    int w = desc.w, h = desc.h;
    for (int i = 0; i < desc.nMipmaps; i++)
    {
        size += MipmapSizeVulkan(format, w, h);
        w = (w > 1) ? w / 2 : 1;
        h = (h > 1) ? h / 2 : 1;
    }
    return size;
}

int SurfaceInfoVulkan::CreateSurface(EngineVulkan* engine, const TextureDescVulkan& desc, PacFormat format,
                                     int totalSize)
{
    _w = desc.w;
    _h = desc.h;
    _nMipmaps = desc.nMipmaps;
    _format = format;
    _id = _nextId++;

    _totalSize = CalculateSize(desc, format, totalSize);
    _usedSize = _totalSize;

    VulkanContext& vk = engine->Context();

    // TransferSrc included for the dynamic-texture mip-generation blits.
    vk::ImageCreateInfo info({}, vk::ImageType::e2D, desc.vkFormat, vk::Extent3D((uint32_t)desc.w, (uint32_t)desc.h, 1),
                             (uint32_t)desc.nMipmaps, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
                             vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst |
                                 vk::ImageUsageFlagBits::eTransferSrc);
    VkImageCreateInfo rawInfo = info;
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    VkImage rawImage = VK_NULL_HANDLE;
    if (vmaCreateImage(vk.allocator, &rawInfo, &allocInfo, &rawImage, &_alloc, nullptr) != VK_SUCCESS)
    {
        LOG_ERROR(Graphics, "VK: texture image allocation failed ({}x{} mips={})", desc.w, desc.h, desc.nMipmaps);
        return -1;
    }
    _image = rawImage;

    // PacAI88 (D3D A8L8) samples as (L,L,L,A); R8G8 gives (R,G,0,1) —
    // replicate via view swizzle (GL33 does the same with texture swizzle).
    vk::ComponentMapping components;
    if (format == PacAI88)
        components = {vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eR,
                      vk::ComponentSwizzle::eG};

    try
    {
        _view = vk.device.createImageView(
            {{},
             _image,
             vk::ImageViewType::e2D,
             desc.vkFormat,
             components,
             vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, (uint32_t)desc.nMipmaps, 0, 1)});
    }
    catch (const std::exception& e)
    {
        LOG_ERROR(Graphics, "VK: texture view creation failed: {}", e.what());
        vmaDestroyImage(vk.allocator, rawImage, _alloc);
        _image = nullptr;
        _alloc = nullptr;
        return -1;
    }

    return 0;
}

void SurfaceInfoVulkan::Free(bool lastRef, int)
{
    if (lastRef && _image)
        static_cast<EngineVulkan*>(GEngine)->DeferDestroyImage(_image, _alloc, _view);
    _image = nullptr;
    _view = nullptr;
    _alloc = nullptr;
    _totalSize = 0;
    _usedSize = 0;
    _w = 0;
    _h = 0;
    _nMipmaps = 0;
}

void SurfaceInfoVulkan::TakeOver(SurfaceInfoVulkan& from)
{
    *this = from;
    from.Free(false);
}

// ── Dynamic (raw RGBA) textures — font atlases, overlays ───────────────────

namespace
{

void CmdImageBarrier(vk::CommandBuffer cmd, vk::Image image, vk::ImageLayout from, vk::ImageLayout to,
                     vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess, vk::PipelineStageFlags2 dstStage,
                     vk::AccessFlags2 dstAccess, uint32_t baseMip, uint32_t mipCount)
{
    vk::ImageMemoryBarrier2 barrier(srcStage, srcAccess, dstStage, dstAccess, from, to, vk::QueueFamilyIgnored,
                                    vk::QueueFamilyIgnored, image,
                                    vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, baseMip, mipCount, 0, 1));
    cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, barrier));
}

// Level 0 must be TransferDst with fresh content; leaves ALL levels ShaderReadOnly.
void GenerateMipChain(vk::CommandBuffer cmd, vk::Image image, int w, int h, int mipLevels)
{
    int srcW = w, srcH = h;
    for (int level = 1; level < mipLevels; ++level)
    {
        // Previous level: TransferDst -> TransferSrc
        CmdImageBarrier(cmd, image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eTransferSrcOptimal,
                        vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferWrite,
                        vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferRead, level - 1, 1);

        const int dstW = srcW > 1 ? srcW / 2 : 1;
        const int dstH = srcH > 1 ? srcH / 2 : 1;
        vk::ImageBlit region(vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, level - 1, 0, 1),
                             {vk::Offset3D(0, 0, 0), vk::Offset3D(srcW, srcH, 1)},
                             vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, level, 0, 1),
                             {vk::Offset3D(0, 0, 0), vk::Offset3D(dstW, dstH, 1)});
        cmd.blitImage(image, vk::ImageLayout::eTransferSrcOptimal, image, vk::ImageLayout::eTransferDstOptimal, region,
                      vk::Filter::eLinear);
        srcW = dstW;
        srcH = dstH;
    }

    if (mipLevels > 1)
        CmdImageBarrier(cmd, image, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
                        vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferRead,
                        vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead, 0,
                        mipLevels - 1);
    CmdImageBarrier(cmd, image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferWrite,
                    vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead,
                    mipLevels - 1, 1);
}

} // namespace

bool TextureVulkan::InitFromRGBA(int w, int h, const void* rgba, uint32_t size, bool mipmap)
{
    if (!rgba)
        return false;

    _initialized = true;
    _dynamicMipmapped = mipmap;

    int maxLevel = 0;
    if (mipmap)
    {
        int dim = w > h ? w : h;
        while (dim > 1)
        {
            dim >>= 1;
            maxLevel++;
        }
    }
    _nMipmaps = maxLevel + 1;
    _mipmaps[0]._w = static_cast<short>(w);
    _mipmaps[0]._h = static_cast<short>(h);
    _mipmaps[0]._pitch = static_cast<short>(w * 4);
    _maxSize = w > h ? w : h;

    EngineVulkan* engine = static_cast<EngineVulkan*>(GEngine);

    TextureDescVulkan desc;
    desc.w = w;
    desc.h = h;
    desc.nMipmaps = _nMipmaps;
    desc.vkFormat = vk::Format::eR8G8B8A8Unorm; // raw RGBA byte order
    desc.compressed = false;

    // PacARGB8888 as the bookkeeping format: byte size matches (w*h*4).
    if (_surface.CreateSurface(engine, desc, PacARGB8888, w * h * 4) < 0)
        return false;
    _surface._nMipmaps = 1; // GL33 quirk kept: bookkeeping says 1, sampling uses all

    UpdateRGBA(rgba, size);

    _levelLoaded = 0;
    _smallLoaded = 0;
    _largestUsed = 0;

    return true;
}

void TextureVulkan::UpdateRGBA(const void* rgba, uint32_t /*size*/)
{
    if (!_surface.HasImage() || !rgba)
        return;

    EngineVulkan* engine = static_cast<EngineVulkan*>(GEngine);
    const int w = _surface._w;
    const int h = _surface._h;
    const vk::DeviceSize bytes = (vk::DeviceSize)w * h * 4;
    const int mipLevels = _dynamicMipmapped ? _nMipmaps : 1;

    EngineVulkan::UploadTicket ticket = engine->BeginTextureUpload();
    vk::Buffer stagingBuf;
    vk::DeviceSize stagingOffset = 0;
    uint8_t* dst = engine->AllocStaging(ticket, bytes, stagingBuf, stagingOffset);
    if (!dst)
    {
        engine->EndTextureUpload(ticket);
        return;
    }
    memcpy(dst, rgba, bytes);

    // Whole image: discard old content (Undefined source layout). AllCommands
    // as src stage orders against in-flight reads of the previous content.
    CmdImageBarrier(ticket.cmd, _surface.GetImage(), vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
                    vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlags2{}, vk::PipelineStageFlagBits2::eCopy,
                    vk::AccessFlagBits2::eTransferWrite, 0, mipLevels);

    vk::BufferImageCopy region(stagingOffset, 0, 0, vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),
                               {0, 0, 0}, {(uint32_t)w, (uint32_t)h, 1});
    ticket.cmd.copyBufferToImage(stagingBuf, _surface.GetImage(), vk::ImageLayout::eTransferDstOptimal, region);

    if (_dynamicMipmapped && mipLevels > 1)
    {
        GenerateMipChain(ticket.cmd, _surface.GetImage(), w, h, mipLevels);
    }
    else
    {
        CmdImageBarrier(ticket.cmd, _surface.GetImage(), vk::ImageLayout::eTransferDstOptimal,
                        vk::ImageLayout::eShaderReadOnlyOptimal, vk::PipelineStageFlagBits2::eCopy,
                        vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eFragmentShader,
                        vk::AccessFlagBits2::eShaderSampledRead, 0, mipLevels);
    }

    engine->EndTextureUpload(ticket);
}

} // namespace Poseidon
