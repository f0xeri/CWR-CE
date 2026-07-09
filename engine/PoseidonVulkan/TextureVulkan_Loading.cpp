// Mirror of PoseidonGL33/TextureGL33_Loading.cpp. The gl(Compressed)SubImage2D
// sequence becomes: staging-arena write -> vkCmdCopyBufferToImage into the
// frame's upload command buffer -> barrier to shader-read. GL's dedicated
// upload unit trick has no analogue: uploads never touch binding state here.
#include <Poseidon/Core/Application.hpp>

#include "TextureVulkan.hpp"
#include "EngineVulkan.hpp"
#include <Poseidon/IO/FileServer.hpp>
#include <Poseidon/IO/Streams/QBStream.hpp>

#include <Poseidon/Graphics/Core/MipmapLayout.hpp>

namespace Poseidon
{

namespace
{
bool IsCompressedInterpolationFormat(PacFormat format)
{
    switch (format)
    {
        case PacDXT1:
        case PacDXT2:
        case PacDXT3:
        case PacDXT4:
        case PacDXT5:
            return true;
        default:
            return false;
    }
}

PacFormat UploadFormatForTextureVulkan(PacFormat format, bool interpolate)
{
    if (interpolate && IsCompressedInterpolationFormat(format))
        return PacARGB1555;
    return format;
}

} // namespace

void InitVkPixelFormat(TextureDescVulkan& desc, PacFormat format); // TextureVulkan_Init.cpp

void TextureVulkan::InitDesc(TextureDescVulkan& desc, int levelMin)
{
    memset(&desc, 0, sizeof(desc));

    PacFormat format = UploadFormatForTextureVulkan(_mipmaps[levelMin].DstFormat(), _interpolate);
    InitVkPixelFormat(desc, format);

    desc.w = _mipmaps[levelMin]._w;
    desc.h = _mipmaps[levelMin]._h;
    desc.nMipmaps = _nMipmaps - levelMin;
}

int TextureVulkan::TotalSize(int levelMin) const
{
    int totalSize = 0;
    for (int i = levelMin; i < _nMipmaps; i++)
    {
        const PacLevelMem& mip = _mipmaps[i];
        totalSize += MipmapSizeVulkan(UploadFormatForTextureVulkan(_mipmaps[levelMin].DstFormat(), _interpolate),
                                      mip._w, mip._h);
    }
    return totalSize;
}

int TextureVulkan::UploadToGPU(SurfaceInfoVulkan& surface, int levelMin)
{
    if (!_src)
    {
        RptF("No texture source for %s", Name());
        return -1;
    }

    if (!surface.HasImage())
        return -1;

    EngineVulkan* engine = static_cast<EngineVulkan*>(GEngine);
    EngineVulkan::UploadTicket ticket = engine->BeginTextureUpload();

    const int nLevels = _nMipmaps - levelMin;

    // Whole image goes TransferDst, discarding previous content (matters for
    // reused surfaces). AllCommands src stage orders against in-flight frames
    // still sampling the reused image.
    {
        vk::ImageMemoryBarrier2 toDst(vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlags2{},
                                      vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite,
                                      vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
                                      vk::QueueFamilyIgnored, vk::QueueFamilyIgnored, surface.GetImage(),
                                      vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, nLevels, 0, 1));
        ticket.cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, toDst));
    }

    // See TextureGL33::UploadToGPU: level-min format is authoritative for the
    // whole chain (the image was allocated with it; mixed-format PAAs exist).
    const PacFormat sharedFmt = UploadFormatForTextureVulkan(_mipmaps[levelMin].DstFormat(), _interpolate);

    int ret = 0;
    for (int i = levelMin; i < _nMipmaps; i++)
    {
        PacLevelMem& srcMip = _mipmaps[i];
        PacLevelMem mip = srcMip;
        if (mip.DstFormat() != sharedFmt)
            mip.SetDestFormat(sharedFmt, 8);
        const int aLevel = i - levelMin;

        // Tight per-mip layout — same helper the GL33 loader uses (I-15/I-16).
        const auto layout = render::mipmap::ComputeLayout(sharedFmt, srcMip._w, srcMip._h);
        const int dataSize = layout.dataSize;

        vk::Buffer stagingBuf;
        vk::DeviceSize stagingOffset = 0;
        uint8_t* pixelData = engine->AllocStaging(ticket, dataSize, stagingBuf, stagingOffset);
        if (!pixelData)
        {
            ret = -1;
            break;
        }

        int loaded = _src->GetMipmapData(reinterpret_cast<char*>(pixelData), mip, i);

        if (_interpolate)
        {
            PoseidonAssert(_interpolate->_nMipmaps == _nMipmaps);
            PacLevelMem imip = _interpolate->_mipmaps[i];
            if (imip.DstFormat() != sharedFmt)
                imip.SetDestFormat(sharedFmt, 8);

            AUTO_STATIC_ARRAY(char, imem, 256 * 256 * 4);
            imem.Realloc(dataSize);
            imem.Resize(dataSize);

            _interpolate->_src->GetMipmapData(imem.Data(), imip, i);
            mip.Interpolate(reinterpret_cast<char*>(pixelData), imem.Data(), imip, _iFactor);
        }

        if (!loaded)
        {
            memset(pixelData, 0, dataSize);
            Foundation::WarningMessage("Cannot load mipmap %s", Name());
        }

        vk::BufferImageCopy region(stagingOffset, 0, 0,
                                   vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, aLevel, 0, 1), {0, 0, 0},
                                   {(uint32_t)srcMip._w, (uint32_t)srcMip._h, 1});
        ticket.cmd.copyBufferToImage(stagingBuf, surface.GetImage(), vk::ImageLayout::eTransferDstOptimal, region);
    }

    {
        vk::ImageMemoryBarrier2 toRead(vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite,
                                       vk::PipelineStageFlagBits2::eFragmentShader,
                                       vk::AccessFlagBits2::eShaderSampledRead, vk::ImageLayout::eTransferDstOptimal,
                                       vk::ImageLayout::eShaderReadOnlyOptimal, vk::QueueFamilyIgnored,
                                       vk::QueueFamilyIgnored, surface.GetImage(),
                                       vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, nLevels, 0, 1));
        ticket.cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, toRead));
    }

    engine->EndTextureUpload(ticket);
    return ret;
}

int TextureVulkan::LoadLevels(int levelMin)
{
    if (levelMin < 0)
        return 0;

    TextBankVulkan* bank = static_cast<TextBankVulkan*>(GEngine->TextBank());

    PoseidonAssert(levelMin < _nMipmaps);
    PoseidonAssert(levelMin >= 0);

    int ret = 0;

    if (_interpolate)
        _interpolate->_inUse++;

    if (_levelLoaded > levelMin)
    {
        ReleaseMemory(true);

        _inUse++;

        TextureDescVulkan desc;
        InitDesc(desc, levelMin);

        PacFormat format = UploadFormatForTextureVulkan(_mipmaps[levelMin].DstFormat(), _interpolate);
        bank->UseReleased(_surface, desc, format);

        if (!_surface.HasImage())
        {
            if (bank->_totalAllocated > bank->_limitAllocatedTextures - 512 * 1024)
                bank->Reuse(_surface, desc, format);
        }

        int totalSize = TotalSize(levelMin);

        if (!_surface.HasImage())
        {
            bank->ReserveMemory(totalSize);

            if (bank->CreateGPUSurface(_surface, desc, format, totalSize) < 0)
            {
                _inUse--;
                if (_interpolate)
                    _interpolate->_inUse--;
                return -1;
            }

            bank->_thisFrameAlloc++;
        }

        ret = UploadToGPU(_surface, levelMin);

        _inUse--;

        CacheUse(bank->_thisFrameWholeUsed);
        _levelLoaded = levelMin;
    }

    if (_interpolate)
        _interpolate->_inUse--;

    if (ret < 0)
        ReleaseMemory(true);

    return ret;
}

void TextureVulkan::ReleaseSmall(bool store)
{
    TextBankVulkan* bank = static_cast<TextBankVulkan*>(GEngine->TextBank());
    if (_smallSurface.HasImage())
    {
        if (store)
        {
            bank->AddReleased(_smallSurface);
            _smallSurface.Free(false);
        }
        else
        {
            bank->_totalAllocated -= _smallSurface.SizeUsed();
            bank->_thisFrameAlloc++;
            _smallSurface.Free(true);
        }
        _smallLoaded = _nMipmaps;
    }
}

int TextureVulkan::LoadSmall()
{
    if (_smallSurface.HasImage())
        return 0;

    TextBankVulkan* bank = static_cast<TextBankVulkan*>(GEngine->TextBank());

    if (_nMipmaps <= 0)
        return -1;

    int i;
    for (i = 0; i < _nMipmaps; i++)
    {
        int pixels = _mipmaps[i]._w * _mipmaps[i]._h;
        if (pixels <= bank->_maxSmallTexturePixels)
            break;
    }
    int levelMin = i;
    if (levelMin >= _nMipmaps)
        levelMin = _nMipmaps - 1;

    PacFormat format = UploadFormatForTextureVulkan(_mipmaps[levelMin].DstFormat(), _interpolate);
    TextureDescVulkan desc;
    InitDesc(desc, levelMin);

    bank->UseReleased(_smallSurface, desc, format);

    int totalSize = TotalSize(levelMin);

    if (!_smallSurface.HasImage())
    {
        bank->ReserveMemory(totalSize);

        if (bank->CreateGPUSurface(_smallSurface, desc, format, totalSize) < 0)
            return -1;
    }

    int ret = UploadToGPU(_smallSurface, levelMin);
    if (ret >= 0)
    {
        _smallLoaded = levelMin;
        return 0;
    }
    return -1;
}

void TextureVulkan::MemoryReleased()
{
    if (_cache)
    {
        _cache->Delete();
        delete _cache;
        _cache = nullptr;
    }
    _levelLoaded = _nMipmaps;
}

void TextureVulkan::ReleaseMemory(bool store)
{
    TextBankVulkan* bank = static_cast<TextBankVulkan*>(GEngine->TextBank());
    if (_surface.HasImage())
    {
        if (store)
        {
            bank->AddReleased(_surface);
        }
        else
        {
            bank->_totalAllocated -= _surface.SizeUsed();
            bank->_thisFrameAlloc++;
        }
        _surface.Free(!store);
        MemoryReleased();
    }
}

void TextureVulkan::ReuseMemory(SurfaceInfoVulkan& surf)
{
    if (_surface.HasImage())
    {
        surf.TakeOver(_surface);
    }
    MemoryReleased();
}

void TextureVulkan::CacheUse(VulkanMipCacheRoot& list)
{
    HMipCacheVulkan* first;
    if (_cache)
    {
        _cache->Delete();
        first = _cache;
    }
    else
    {
        first = new HMipCacheVulkan;
    }
    first->texture = this;
    list.Insert(first);
    _cache = first;
}

} // namespace Poseidon
