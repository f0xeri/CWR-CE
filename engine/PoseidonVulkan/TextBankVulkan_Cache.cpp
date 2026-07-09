// Mirror of PoseidonGL33/TextureBankGL33_Cache.cpp: per-frame LRU bookkeeping,
// the UseMipmap demand-streaming heart, detail-texture init, VRAM budget via
// VMA heap budgets (replacing the GL vendor extensions).
#include <Poseidon/IO/ParamFile/ParamFile.hpp>

extern ParamFile Remaster;

#include <Poseidon/IO/Streams/QBStream.hpp>
#include <Poseidon/Foundation/Algorithms/Qsort.hpp>
#include "TextureVulkan.hpp"
#include "EngineVulkan.hpp"
#include <Poseidon/Core/Progress.hpp>
#include <Poseidon/Core/Global.hpp>

#include <Poseidon/Input/InputSubsystem.hpp>

namespace Poseidon
{

void TextBankVulkan::StartFrame()
{
    InitDetailTextures();
    CheckTextureMemory();
    _thisFrameCopied = 0;
    _thisFrameAlloc = 0;
    if (_loadBoostFrames > 0)
        --_loadBoostFrames;

#if _ENABLE_CHEATS
    if (InputSubsystem::Instance().GetCheat2ToDo(SDL_SCANCODE_E))
    {
        ReportTextures("textures.txt");
    }
#endif
}

void TextBankVulkan::FinishFrame()
{
    for (HMipCacheVulkan* tc = _thisFramePartialUsed.Last(); tc; tc = _thisFramePartialUsed.Prev(tc))
    {
        TextureVulkan* tex = tc->texture;
        tex->_levelNeededLastFrame = tex->_levelNeededThisFrame;
        tex->_levelNeededThisFrame = tex->_nMipmaps;
        tex->ResetMipmap();
    }
    for (HMipCacheVulkan* tc = _thisFrameWholeUsed.Last(); tc; tc = _thisFrameWholeUsed.Prev(tc))
    {
        TextureVulkan* tex = tc->texture;
        tex->_levelNeededLastFrame = tex->_levelNeededThisFrame;
        tex->_levelNeededThisFrame = tex->_nMipmaps;
        tex->ResetMipmap();
    }
    for (HMipCacheVulkan* tc = _lastFramePartialUsed.Last(); tc; tc = _lastFramePartialUsed.Prev(tc))
    {
        TextureVulkan* tex = tc->texture;
        tex->_levelNeededLastFrame = tex->_levelNeededThisFrame;
        tex->_levelNeededThisFrame = tex->_nMipmaps;
        tex->ResetMipmap();
    }
    for (HMipCacheVulkan* tc = _lastFrameWholeUsed.Last(); tc; tc = _lastFrameWholeUsed.Prev(tc))
    {
        TextureVulkan* tex = tc->texture;
        tex->_levelNeededLastFrame = tex->_levelNeededThisFrame;
        tex->_levelNeededThisFrame = tex->_nMipmaps;
        tex->ResetMipmap();
    }

    _previousUsed.Move(_lastFrameWholeUsed);
    PoseidonAssert(_lastFrameWholeUsed.Last() == nullptr);

    _lastFrameWholeUsed.Move(_thisFrameWholeUsed);
    PoseidonAssert(_thisFrameWholeUsed.Last() == nullptr);

    _previousUsed.Move(_lastFramePartialUsed);
    PoseidonAssert(_lastFramePartialUsed.Last() == nullptr);

    _lastFramePartialUsed.Move(_thisFramePartialUsed);
    PoseidonAssert(_thisFramePartialUsed.Last() == nullptr);
}

const int MaxAllocationsPerFrame = 8;
const int MaxCopyPerFrame = 32768;

MipInfo TextBankVulkan::UseMipmap(Texture* absTexture, int level, int top)
{
    if (!absTexture)
        return MipInfo(nullptr, 0);

    TextureVulkan* texture = static_cast<TextureVulkan*>(absTexture);
    texture->LoadHeadersNV();

    // Dynamic textures (CreateDynamic — font atlases, etc.) have no _src to
    // demand-load from; their content is already resident. See the GL33
    // comment for the garbage-sampling failure mode this avoids.
    if (!texture->_src && texture->_surface.HasImage())
        return MipInfo(texture, 0);

    saturateMin(level, texture->_mipmapNeeded);
    saturateMin(top, texture->_mipmapWanted);

    if (level < 0)
        level = 0;

    saturateMin(level, texture->_nMipmaps - 1);
    saturateMax(top, texture->_largestUsed);
    saturateMin(top, level);
    saturateMax(level, top);

    // Never use mipmaps smaller than some limit
    int limitUse = _maxSmallTexturePixels / 4;
    for (; level > 0; level--)
    {
        PacLevelMem* mipTop = &texture->_mipmaps[level];
        if (mipTop->_w * mipTop->_h >= limitUse)
            break;
    }

    saturateMin(top, level);

    // Budget gate is lifted while a load boost is active.
    if (_loadBoostFrames <= 0 && (_thisFrameCopied > MaxCopyPerFrame || _thisFrameAlloc > MaxAllocationsPerFrame))
    {
        if (texture->_levelLoaded < texture->_nMipmaps)
        {
            top = level = texture->_levelLoaded;
        }
        else if (texture->_smallLoaded < texture->_nMipmaps)
        {
            top = level = texture->_smallLoaded;
        }
    }

    // Use small texture if adequate
    if (texture->_smallLoaded <= level)
    {
        if (texture->LoadSmall() < 0)
            return MipInfo(texture, -1);
        return MipInfo(texture, texture->_smallLoaded);
    }

    if (texture->_levelLoaded > level)
    {
        level = top;

        if (texture->_levelNeededThisFrame > level)
            texture->_levelNeededThisFrame = level;

        for (;;)
        {
            int ret = texture->LoadLevels(level);
            if (ret >= 0)
                break;
            LOG_DEBUG(Graphics, "VK: Out of VID: Try next level");
            level++;
            if (level >= texture->_nMipmaps)
                break;
        }
    }
    else
    {
        if (texture->_levelNeededThisFrame > level)
            texture->_levelNeededThisFrame = level;

        PoseidonAssert(texture->_cache);
        if (texture->LevelNeeded() <= texture->_levelLoaded)
        {
            texture->CacheUse(_thisFrameWholeUsed);
        }
        else
        {
            texture->CacheUse(_thisFramePartialUsed);
        }
    }

    level = texture->_levelLoaded;
    if (level >= texture->_nMipmaps)
    {
        if (texture->LoadSmall() < 0)
            return MipInfo(texture, -1);
        return MipInfo(texture, texture->_smallLoaded);
    }

    return MipInfo(texture, level);
}

void TextBankVulkan::InitDetailTextures()
{
    if (_detail)
        return;

    const ParamEntry& names = Remaster >> "CfgDetailTextures";
    RStringB detailName = names >> "detail";
    if (QIFStreamB::FileExist(detailName))
    {
        _detail = new TextureVulkan;
        _detail->Init(detailName);
        _detail->_isDetail = true;
    }

    RStringB specularName = names >> "specular";
    if (QIFStreamB::FileExist(specularName))
    {
        _specular = new TextureVulkan;
        _specular->Init(specularName);
        _specular->_isDetail = true;
    }

    RStringB grassName = names >> "grass";
    if (QIFStreamB::FileExist(grassName))
    {
        _grass = new TextureVulkan;
        _grass->Init(grassName);
        _grass->SetMaxSize(1024);
        _grass->_isDetail = true;
    }

    RStringB waterName = names >> "waterBump";
    if (QIFStreamB::FileExist(waterName))
    {
        _waterBump = new TextureVulkan;
        _waterBump->Init(waterName);
        _waterBump->SetMaxSize(1024);
        _waterBump->_isDetail = true;
    }
}

void TextBankVulkan::FlushTextures()
{
    // No-op, same contract as GL33 (see its comment).
}

void TextBankVulkan::ForceReloadAll()
{
    int dropped = 0;
    int skipped = 0;
    for (int i = 0; i < _texture.Size(); i++)
    {
        TextureVulkan* tex = _texture[i];
        if (!tex)
            continue;
        if (!tex->_src)
        {
            ++skipped;
            continue;
        }
        tex->ReleaseMemory(false);
        tex->ReleaseSmall(false);
        tex->_src = nullptr;
        tex->_initialized = false;
        tex->_levelLoaded = MAX_MIPMAPS;
        tex->_smallLoaded = MAX_MIPMAPS;
        tex->_levelNeededThisFrame = MAX_MIPMAPS;
        tex->_levelNeededLastFrame = MAX_MIPMAPS;
        ++dropped;
    }
    LOG_INFO(Graphics, "TextBankVulkan::ForceReloadAll: dropped {} textures, kept {} dynamic", dropped, skipped);
}

void TextBankVulkan::FlushBank(QFBank* bank)
{
    for (int i = 0; i < _texture.Size(); i++)
    {
        TextureVulkan* tex = _texture[i];
        if (!tex)
            continue;
        if (!bank->FileExists(tex->GetName()))
            continue;
        _texture.Delete(i);
        i--;
    }
}

static int CompareTextureFileOrderVulkan(const LLink<TextureVulkan>* tl1, const LLink<TextureVulkan>* tl2)
{
    TextureVulkan* t1 = *tl1;
    TextureVulkan* t2 = *tl2;
    const char* n1 = t1->GetName();
    const char* n2 = t2->GetName();
    QFBank* b1 = QIFStreamB::AutoBank(t1->GetName());
    QFBank* b2 = QIFStreamB::AutoBank(t2->GetName());
    if (b1 > b2)
        return -1;
    if (b1 < b2)
        return +1;
    PoseidonAssert(b1 == b2);
    if (!b1)
        return 0;
    int o1 = b1->GetFileOrder(n1 + strlen(b1->GetPrefix()));
    int o2 = b2->GetFileOrder(n2 + strlen(b2->GetPrefix()));
    return o1 - o2;
}

void TextBankVulkan::Preload()
{
    Compact();

    DWORD start = Foundation::GlobalTickCount();
    QSort(_texture.Data(), _texture.Size(), CompareTextureFileOrderVulkan);

    for (int i = 0; i < _texture.Size(); i++)
    {
        TextureVulkan* tex = _texture[i];
        if (!tex)
            continue;
        tex->LoadHeadersNV();
        ProgressRefresh();
    }
    DWORD end = Foundation::GlobalTickCount();

    LOG_DEBUG(Graphics, "VK: Preload {} textures - {} ms", _texture.Size(), end - start);
}

int TextBankVulkan::FreeTextureMemory()
{
    constexpr int64_t MinSaneMem = 32ll * 1024 * 1024; // 32 MB — no real GPU has less
    // Cap at 512 MB — engine doesn't need more, avoids int overflow with large VRAM
    constexpr int64_t MaxBudget = 512ll * 1024 * 1024;

    // VMA aggregates VK_EXT_memory_budget when available, heap sizes otherwise.
    VmaBudget budgets[VK_MAX_MEMORY_HEAPS] = {};
    vmaGetHeapBudgets(_engine->Context().allocator, budgets);

    const vk::PhysicalDeviceMemoryProperties memProps = _engine->Context().physicalDevice.getMemoryProperties();
    int64_t deviceLocalBudget = 0;
    int64_t deviceLocalUsage = 0;
    for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i)
    {
        if (memProps.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal)
        {
            deviceLocalBudget += (int64_t)budgets[i].budget;
            deviceLocalUsage += (int64_t)budgets[i].usage;
        }
    }

    int64_t available = deviceLocalBudget - deviceLocalUsage;
    if (deviceLocalBudget < MinSaneMem)
    {
        LOG_DEBUG(Graphics, "VK: Using fallback VRAM budget: 256 MB");
        return 256 * 1024 * 1024;
    }
    if (available < 0)
        available = 0;
    if (available > MaxBudget)
        available = MaxBudget;
    return static_cast<int>(available);
}

void TextBankVulkan::CheckTextureMemory()
{
    int freeMem = FreeTextureMemory();
    _limitAllocatedTextures = _totalAllocated + freeMem;
    _limitAllocatedTextures -= _reserveTextureMemory;
}

} // namespace Poseidon
