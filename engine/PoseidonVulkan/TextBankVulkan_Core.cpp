// Mirror of PoseidonGL33/TextureBankGL33_Core.cpp. VRAM budget detection uses
// VMA heap budgets instead of GL vendor extensions; everything else — surface
// free-list, LRU eviction, interpolated loads, dynamic textures — is 1:1.
#include <Poseidon/Core/Application.hpp>

#include <Poseidon/Dev/Diag/ScopedTimer.hpp>
#include <Poseidon/IO/Streams/QBStream.hpp>
#include "TextureVulkan.hpp"
#include "EngineVulkan.hpp"
#include <Poseidon/Graphics/Textures/LooseTextures.hpp>
#include <Poseidon/Foundation/Math/Statistics.hpp>
#include <Poseidon/Core/Global.hpp>

namespace Poseidon
{

static const char* FormatNameVulkan(PacFormat format)
{
    switch (format)
    {
        case PacP8:
            return "PacP8";
        case PacAI88:
            return "PacAI88";
        case PacRGB565:
            return "PacRGB565";
        case PacARGB1555:
            return "PacARGB1555";
        case PacARGB4444:
            return "PacARGB4444";
        case PacARGB8888:
            return "PacARGB8888";
        case PacDXT1:
            return "PacDXT1";
        case PacDXT3:
            return "PacDXT3";
        case PacDXT5:
            return "PacDXT5";
        default:
            return "???";
    }
}

static void SurfaceName(char* buf, int bufSize, const SurfaceInfoVulkan& surface)
{
    snprintf(buf, bufSize, "\t%s,%d,%d,", FormatNameVulkan(surface._format), surface._w, surface._h);
}

TextBankVulkan::TextBankVulkan(EngineVulkan* engine) : _totalAllocated(0)
{
    _reserveTextureMemory = 0;
    _engine = engine;

    CheckTextureMemory();

    _maxTextureMemory = FreeTextureMemory();

    // Small texture limit: use ~1/8 of texture memory for small textures
    int limitPixels = _limitAllocatedTextures / (2 * 1024 * 8);
    int limitPixelsPow2 = 1;
    while (limitPixelsPow2 + limitPixelsPow2 < limitPixels)
    {
        limitPixelsPow2 += limitPixelsPow2;
    }

    _maxSmallTexturePixels = limitPixelsPow2;
    saturateMax(_maxSmallTexturePixels, 4 * 4);
    LOG_DEBUG(Graphics, "VK Max small texture pixels: {}, {:.0f}", _maxSmallTexturePixels,
              sqrt(_maxSmallTexturePixels));

    _memProbe.Register(
        "Textures", 0.5f, [this] { return (size_t)_totalAllocated; }, [this] { return (size_t)_maxTextureMemory; },
        [this] { return (size_t)_texture.Size(); });
}

TextBankVulkan::~TextBankVulkan()
{
    UnlockAllTextures();
    DeleteAllAnimated();
    _texture.Compact();
    PoseidonAssert(_texture.Size() == 0);
    _texture.Clear();
    _detail.Free();
    _waterBump.Free();
    _specular.Free();
    _grass.Free();

    for (int i = 0; i < _freeTextures.Size(); i++)
    {
        _freeTextures[i].Free(true);
    }
    _freeTextures.Clear();
}

bool TextBankVulkan::NativeFormatSupported(PacFormat format) const
{
    const vk::Format vkFormat = PacToVkFormat(format);
    if (vkFormat == vk::Format::eUndefined)
        return false;
    const vk::FormatProperties props = _engine->Context().physicalDevice.getFormatProperties(vkFormat);
    constexpr auto required = vk::FormatFeatureFlagBits::eSampledImage | vk::FormatFeatureFlagBits::eTransferDst;
    return (props.optimalTilingFeatures & required) == required;
}

void TextBankVulkan::Compact()
{
    _texture.Compact();
}

void TextBankVulkan::StopAll() {}

bool TextBankVulkan::VerifyChecksums()
{
    StatisticsByName stats;
    for (int i = 0; i < _freeTextures.Size(); i++)
    {
        const SurfaceInfoVulkan& surface = _freeTextures[i];
        char name[80];
        SurfaceName(name, sizeof(name), surface);
        stats.Count(name);
    }
    LOG_DEBUG(Graphics, "VK Unused texture surfaces");
    stats.Report();
    stats.Clear();

    LOG_DEBUG(Graphics, "VK Used texture surfaces");
    for (int i = 0; i < _texture.Size(); i++)
    {
        TextureVulkan* texture = _texture[i];
        if (!texture)
            continue;
        if (texture->_smallSurface.HasImage())
        {
            char name[80];
            SurfaceName(name, sizeof(name), texture->_smallSurface);
            stats.Count(name);
        }
        if (texture->_surface.HasImage())
        {
            char name[80];
            SurfaceName(name, sizeof(name), texture->_surface);
            stats.Count(name);
        }
    }
    stats.Report();
    stats.Clear();

    return true;
}

int TextBankVulkan::Find(RStringB name1, TextureVulkan* interpolate)
{
    for (int i = 0; i < _texture.Size(); i++)
    {
        TextureVulkan* texture = _texture[i];
        if (texture)
        {
            if (name1 != texture->GetName())
                continue;
            if (texture->_interpolate != interpolate)
                continue;
            return i;
        }
    }
    return -1;
}

int TextBankVulkan::FindFree()
{
    for (int i = 0; i < _texture.Size(); i++)
    {
        TextureVulkan* texture = _texture[i];
        if (!texture)
            return i;
    }
    return _texture.Size();
}

TextureVulkan* TextBankVulkan::Copy(int from)
{
    if (from < 0)
        return nullptr;
    const TextureVulkan* source = _texture[from];
    if (!source)
        return nullptr;
    const char* sName = source->Name();

    int iFree = FindFree();
    PoseidonAssert(iFree >= 0);
    TextureVulkan* texture = new TextureVulkan;

    if (texture->Init(sName))
        return nullptr;

    _texture.Access(iFree);
    _texture[iFree] = texture;
    return texture;
}

Ref<Texture> TextBankVulkan::Load(RStringB name)
{
    int i = Find(name);
    if (i >= 0)
        return _texture[i].GetLink();

    const auto _perfTexLoadStart = ::Poseidon::Dev::Perf::Now();

    int iFree = _texture.Size();

    RString resolved = Graphics::ResolveLooseTexturePath(name);
    if (!QIFStreamB::FileExist(resolved))
    {
        // Same tolerance as GL33: dangling texture refs are endemic in the
        // original data; the caller substitutes the default texture.
        const char* cname = static_cast<const char*>(name);
        if (cname && cname[0] && cname[0] != '#')
            LOG_WARN(Graphics, "VK: Cannot load texture {}", cname);
        else
            LOG_DEBUG(Graphics, "VK: Cannot load texture {}", cname);
        return nullptr;
    }

    Ref<TextureVulkan> texture = new TextureVulkan;
    if (!texture)
        return nullptr;

    if (texture->Init(name))
        return nullptr;

    _texture.Access(iFree);
    TextureVulkan* txt = texture;
    _texture[iFree] = txt;

    const double _perfTexLoadMs = ::Poseidon::Dev::Perf::ElapsedMs(_perfTexLoadStart);
    if (_perfTexLoadMs >= 0.5)
    {
        LOG_DEBUG(Graphics, "PERF: TextBank::Load {} took {:.2f}ms", static_cast<const char*>(name), _perfTexLoadMs);
    }
    ::Poseidon::Dev::Perf::EmitTraceEventAsset(Foundation::LogCategory::Graphics, "TextBank::Load", _perfTexLoadStart,
                                               static_cast<const char*>(name));
    return txt;
}

Ref<Texture> TextBankVulkan::LoadInterpolated(RStringB n1, RStringB n2, float factor)
{
    const float eps = 1.0 / 256;
    if (factor >= 1.0 - eps)
        return Load(n2);
    if (factor <= eps)
        return Load(n1);

    Ref<Texture> txt2 = Load(n2);
    TextureVulkan* interpolate = static_cast<TextureVulkan*>(txt2.GetRef());
    int index = Find(n1, interpolate);
    if (index >= 0)
    {
        TextureVulkan* t = _texture[index];
        const float iPolEps = 1.0 / 64;
        if (fabs(t->_iFactor - factor) > iPolEps)
        {
            t->ReleaseMemory(true);
            t->ReleaseSmall(true);
            t->_iFactor = factor;
        }
        return t;
    }
    Ref<Texture> temp = Load(n1);
    int index1 = Find(n1);
    Ref<TextureVulkan> t = Copy(index1);
    if (t)
    {
        t->_interpolate = interpolate;
        t->_iFactor = factor;
    }
    return t.GetRef();
}

void TextBankVulkan::ReleaseAllTextures()
{
    LOG_DEBUG(Graphics, "VK: Allocated before ReleaseAllTextures {}", _totalAllocated);
    ReserveMemory(_previousUsed, 0);
    ReserveMemory(_lastFramePartialUsed, 0);
    ReserveMemory(_lastFrameWholeUsed, 0);
    ReserveMemory(_thisFramePartialUsed, 0);
    ReserveMemory(_thisFrameWholeUsed, 0);
    LOG_DEBUG(Graphics, "VK: Allocated after ReleaseAllTextures {}", _totalAllocated);
}

bool TextBankVulkan::ReserveMemory(VulkanMipCacheRoot& root, int limit)
{
    bool someReleased = false;
    while (_freeTextures.Size() > 0 && _totalAllocated > limit)
    {
        DeleteLastReleased();
        someReleased = true;
    }
    HMipCacheVulkan* last = root.Last();
    while (_totalAllocated > limit)
    {
        if (!last)
            break;

        TextureVulkan* texture = last->texture;
        if (texture->_inUse)
        {
            last = root.Prev(last);
            continue;
        }

        PoseidonAssert(texture->_cache == last);
        someReleased = true;
        texture->ReleaseMemory(false);
        last = root.Last();
    }
    return someReleased;
}

void TextBankVulkan::ReportTextures(const char* /*name*/) {}

bool TextBankVulkan::ReserveMemory(int size)
{
    if (size >= INT_MAX)
    {
        _reserveTextureMemory = 0;
    }
    return ReserveMemory(_previousUsed, _limitAllocatedTextures - size);
}

bool TextBankVulkan::ForcedReserveMemory(int size)
{
    const int minRelease = 4 * 1024;
    if (size < minRelease)
        size = minRelease;

    int newLimit = _totalAllocated - size;
    LOG_DEBUG(Graphics, "VK: ForcedReserveMemory newLimit={} _totalAllocated={}", newLimit, _totalAllocated);

    bool ret = ReserveMemory(_previousUsed, newLimit);

    if (!ret)
    {
        ret = ReserveMemory(_lastFramePartialUsed, newLimit);
        if (!ret)
        {
            ret = ReserveMemory(_lastFrameWholeUsed, newLimit);
            if (!ret)
            {
                Glob.fullDropDownChange = 0.1;
                LOG_DEBUG(Graphics, "VK: Evicting this frame partially used texture");
                ret = ReserveMemory(_thisFramePartialUsed, newLimit);
                if (!ret)
                {
                    LOG_DEBUG(Graphics, "VK: Evicting this frame whole used texture");
                    ret = ReserveMemory(_thisFrameWholeUsed, newLimit);
                }
            }
        }
    }
    CheckTextureMemory();
    LOG_DEBUG(Graphics, "  done {} _totalAllocated={}", ret, _totalAllocated);
    return ret;
}

int TextBankVulkan::FindSurface(int w, int h, int nMipmaps, PacFormat format,
                                const AutoArray<SurfaceInfoVulkan>& array) const
{
    for (int i = 0; i < array.Size(); i++)
    {
        const SurfaceInfoVulkan& surface = array[i];
        if (nMipmaps != surface._nMipmaps)
            continue;
        if (surface._w != w)
            continue;
        if (surface._h != h)
            continue;
        if (surface._format == format)
            return i;
    }
    return -1;
}

void TextBankVulkan::DeleteLastReleased()
{
    SurfaceInfoVulkan& free = _freeTextures[0];
    int size = free.SizeUsed();
    _totalAllocated -= size;
    _thisFrameAlloc++;
    free.Free(true);
    _freeTextures.Delete(0);
}

void TextBankVulkan::AddReleased(SurfaceInfoVulkan& surf)
{
    _freeTextures.Add(surf);
}

void TextBankVulkan::UseReleased(SurfaceInfoVulkan& surf, const TextureDescVulkan& desc, PacFormat format)
{
    int reuse = FindReleased(desc.w, desc.h, desc.nMipmaps, format);
    if (reuse < 0)
        return;

    SurfaceInfoVulkan& reused = _freeTextures[reuse];
    surf.TakeOver(reused);
    _freeTextures.Delete(reuse);
}

void TextBankVulkan::Reuse(SurfaceInfoVulkan& surf, const TextureDescVulkan& desc, PacFormat format)
{
    int w = desc.w;
    int h = desc.h;
    int nMipmaps = desc.nMipmaps;
    for (HMipCacheVulkan* last = _previousUsed.Last(); last; last = _previousUsed.Prev(last))
    {
        TextureVulkan* texture = last->texture;
        if (texture->_inUse)
            continue;

        const SurfaceInfoVulkan& surface = texture->_surface;
        if (nMipmaps != surface._nMipmaps)
            continue;
        if (surface._w != w)
            continue;
        if (surface._h != h)
            continue;
        if (surface._format != format)
            continue;

        texture->ReuseMemory(surf);
        return;
    }
}

int TextBankVulkan::CreateGPUSurface(SurfaceInfoVulkan& surface, const TextureDescVulkan& desc, PacFormat format,
                                     int totalSize)
{
    int ret = surface.CreateSurface(_engine, desc, format, totalSize);
    if (ret < 0)
    {
        LOG_DEBUG(Graphics, "VK: CreateGPUSurface failed, trying ForcedReserveMemory");
        ForcedReserveMemory(totalSize);
        ret = surface.CreateSurface(_engine, desc, format, totalSize);
        if (ret < 0)
        {
            Foundation::ErrorMessage("VK: Cannot create texture (%dx%d)", desc.w, desc.h);
            return -1;
        }
    }
    _totalAllocated += surface.SizeUsed();
    return 0;
}

Texture* TextBankVulkan::CreateDynamic(int w, int h, const void* rgba, uint32_t size, bool mipmap)
{
    int idx = FindFree();
    TextureVulkan* tex = new TextureVulkan();
    _texture.Access(idx);
    _texture[idx] = tex;
    if (!tex->InitFromRGBA(w, h, rgba, size, mipmap))
    {
        LOG_WARN(Graphics, "VK: Failed to create dynamic texture {}x{}", w, h);
        return nullptr;
    }
    return tex;
}

void TextBankVulkan::UpdateDynamic(Texture* tex, const void* rgba, uint32_t size)
{
    if (!tex || !rgba)
        return;
    static_cast<TextureVulkan*>(tex)->UpdateRGBA(rgba, size);
}

} // namespace Poseidon
