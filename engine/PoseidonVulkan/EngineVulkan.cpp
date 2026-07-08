#include "EngineVulkan.hpp"

#include <Poseidon/Core/Application.hpp>
#include <Poseidon/Core/Config/EngineConfig.hpp>
#include <Poseidon/Graphics/Dummy/TextBankDummy.hpp>
#include <Poseidon/Foundation/Logging/Logging.hpp>
#include <Poseidon/Graphics/Core/ZBiasMath.hpp>
#include <Poseidon/Graphics/Shared/WindowPlacement.hpp>
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

namespace Poseidon
{

EngineVulkan::EngineVulkan(int width, int height, bool windowed, int bpp)
{
    _bank = new TextBankDummy();
    _w = width;
    _h = height;
    _windowed = windowed;
    _windowedRestoreW = width;
    _windowedRestoreH = height;
    _pixelSize = bpp;
    _depthBpp = 24;
    _refreshRate = 60;

    LOG_INFO(Graphics, "VK: Initializing engine — bootstrap {}x{} {}bpp {} before display.cfg/user overrides", _w, _h,
             _pixelSize, _windowed ? "windowed" : "fullscreen");

    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        LOG_ERROR(Graphics, "VK: SDL_Init failed: {}", SDL_GetError());
        return;
    }

    auto& engineCfg = GApp->GetConfig().GetEngineConfig();
    DisplayPlacementInput displayCfg;
    displayCfg.displayMode = engineCfg.displayMode;
    if (windowed && displayCfg.displayMode != "windowed")
        displayCfg.displayMode = "windowed";
    if (!windowed && displayCfg.displayMode == "windowed")
        displayCfg.displayMode = "borderless";
    displayCfg.width = _w;
    displayCfg.height = _h;

    int desktopW = 0, desktopH = 0, desktopRefresh = 0;
    if (const SDL_DisplayMode* dm = SDL_GetDesktopDisplayMode(SDL_GetPrimaryDisplay()))
    {
        desktopW = dm->w;
        desktopH = dm->h;
        desktopRefresh = (int)(dm->refresh_rate + 0.5f);
    }
    const WindowPlacement placement = ResolveWindowPlacement(displayCfg, desktopW, desktopH, desktopRefresh);
    _windowMode = placement.mode;

    // SDL_WINDOW_VULKAN loads the Vulkan loader library and lets SDL_Vulkan_CreateSurface attach a VkSurfaceKHR to this window later
    Uint32 flags = SDL_WINDOW_VULKAN | SDL_WINDOW_HIGH_PIXEL_DENSITY;
    switch (placement.mode)
    {
        case WindowMode::Fullscreen:
        case WindowMode::Borderless:
            flags |= SDL_WINDOW_BORDERLESS;
            break;
        case WindowMode::Windowed:
            flags |= SDL_WINDOW_RESIZABLE;
            break;
    }

    _sdlWindow = SDL_CreateWindow("Poseidon [VK]", placement.width, placement.height, flags);
    if (!_sdlWindow)
    {
        LOG_ERROR(Graphics, "VK: SDL_CreateWindow failed: {}", SDL_GetError());
        return;
    }

    if (placement.mode == WindowMode::Borderless)
    {
#ifndef _WIN32
        SDL_SetWindowFullscreenMode(_sdlWindow, nullptr);
        if (!SDL_SetWindowFullscreen(_sdlWindow, true))
            LOG_WARN(Graphics, "VK: SDL_SetWindowFullscreen(true) failed for borderless startup: {}", SDL_GetError());
#else
        if (placement.posX != WindowPlacement::kCentered)
            SDL_SetWindowPosition(_sdlWindow, placement.posX, placement.posY);
#endif
    }
    else if (placement.posX != WindowPlacement::kCentered)
    {
        SDL_SetWindowPosition(_sdlWindow, placement.posX, placement.posY);
    }

    _w = placement.width;
    _h = placement.height;
    if (placement.refreshHz > 0)
        _refreshRate = placement.refreshHz;

    if (!_vk.Init(_sdlWindow))
    {
        LOG_ERROR(Graphics, "VK: context initialization failed — engine unusable");
        SDL_DestroyWindow(_sdlWindow);
        _sdlWindow = nullptr;
        return;
    }

    int cw = 0, ch = 0;
    SDL_GetWindowSizeInPixels(_sdlWindow, &cw, &ch);
    _w = cw;
    _h = ch;
    LOG_INFO(Graphics, "VK: surface resolved to {}x{} {}", _w, _h, _windowed ? "windowed" : "fullscreen");

    _eventWindow.Attach(_sdlWindow, _w, _h);

    LoadConfig();

    // TODO(vk-phase0): swapchain + per-frame command buffers/sync, then Clear
    // gets a real implementation and the window shows its first pixels.
}

EngineVulkan::~EngineVulkan()
{
    ClearFontCache();
    delete _bank;
    _bank = nullptr;
    _eventWindow.Detach();
    _vk.Shutdown();
    if (_sdlWindow)
    {
        SDL_DestroyWindow(_sdlWindow);
        _sdlWindow = nullptr;
    }
}

RString EngineVulkan::GetDebugName() const
{
    if (_vk.IsValid())
        return RString("Vulkan 1.3 — ") + RString((const char*)_vk.deviceProps.deviceName);
    return "Vulkan 1.3";
}

RString EngineVulkan::GetRendererName() const
{
    return "Vulkan 1.3";
}

// ── Frame cycle ─────────────────────────────────────────────────────────────

void EngineVulkan::InitDraw() {}
void EngineVulkan::FinishDraw() {}
void EngineVulkan::Pause() {}
void EngineVulkan::Restore() {}
void EngineVulkan::Clear(bool, bool, PackedColor) {}
void EngineVulkan::DrawPicture555(unsigned short*) {}

// ── Lighting / atmosphere hooks ─────────────────────────────────────────────

void EngineVulkan::FogColorChanged(const Color&) {}
void EngineVulkan::LightChanged(const Color&, const Color&) {}
void EngineVulkan::NightEffectChanged(float) {}

// ── Display modes ───────────────────────────────────────────────────────────

bool EngineVulkan::SwitchRes(int, int, int)
{
    return false; // TODO swapchain recreation
}

bool EngineVulkan::SwitchRefreshRate(int)
{
    return false;
}

bool EngineVulkan::SetWindowMode(WindowMode)
{
    return false; // TODO needs swapchain recreation on transition
}

void EngineVulkan::HandleEvents()
{
    _eventWindow.HandleEvents();
}

bool EngineVulkan::IsOpen() const
{
    return _eventWindow.IsOpen();
}

void EngineVulkan::SetMouseGrab(bool grab)
{
    _eventWindow.SetMouseGrab(grab);
}

void EngineVulkan::ListResolutions(FindArray<ResolutionInfo>& ret)
{
    ret.Clear();
    if (_windowed)
        return;

    SDL_DisplayID display = _sdlWindow ? SDL_GetDisplayForWindow(_sdlWindow) : SDL_GetPrimaryDisplay();
    if (!display)
        return;

    int count = 0;
    SDL_DisplayMode** modes = SDL_GetFullscreenDisplayModes(display, &count);
    if (!modes)
        return;

    for (int i = 0; i < count; i++)
    {
        ResolutionInfo info;
        info.w = modes[i]->w;
        info.h = modes[i]->h;
        info.bpp = SDL_BITSPERPIXEL(modes[i]->format);
        ret.AddUnique(info);
    }
    SDL_free(modes);
}

void EngineVulkan::ListRefreshRates(FindArray<int>& ret)
{
    ret.Clear();
    if (_windowed)
    {
        ret.Add(0);
        return;
    }

    SDL_DisplayID display = _sdlWindow ? SDL_GetDisplayForWindow(_sdlWindow) : SDL_GetPrimaryDisplay();
    if (!display)
        return;

    int count = 0;
    SDL_DisplayMode** modes = SDL_GetFullscreenDisplayModes(display, &count);
    if (!modes)
        return;

    for (int i = 0; i < count; i++)
    {
        if (modes[i]->w == Width() && modes[i]->h == Height())
            ret.AddUnique(static_cast<int>(modes[i]->refresh_rate));
    }
    SDL_free(modes);
}

// ── Scalar hooks (values mirror GL33 so scene math stays identical) ─────────

bool EngineVulkan::CanZBias() const
{
    // Match GL33/D3D11: software Z-bias is applied by the producer layer.
    return false;
}

bool EngineVulkan::ZBiasExclusion() const
{
    return false;
}

int EngineVulkan::PixelSize() const
{
    return _pixelSize;
}

int EngineVulkan::RefreshRate() const
{
    return _refreshRate;
}

bool EngineVulkan::CanBeWindowed() const
{
    return true;
}

bool EngineVulkan::IsWindowed() const
{
    return _windowMode == WindowMode::Windowed;
}

bool EngineVulkan::IsResizable() const
{
    return _windowMode == WindowMode::Windowed;
}

float EngineVulkan::ZShadowEpsilon() const
{
    return 0.01f;
}

float EngineVulkan::ZRoadEpsilon() const
{
    return 0.005f;
}

float EngineVulkan::ObjMipmapCoef() const
{
    return 1.5f;
}

float EngineVulkan::LandMipmapCoef() const
{
    return 1.0f;
}

bool EngineVulkan::ShadowsFirst() const
{
    return false;
}

bool EngineVulkan::SortByShape() const
{
    return false;
}

int EngineVulkan::GetBias()
{
    return _bias;
}

void EngineVulkan::SetBias(int bias)
{
    _bias = bias;
}

void EngineVulkan::GetZCoefs(float& zAdd, float& zMult)
{
    const auto c = render::zbias::SoftwareCoefs(_bias);
    zMult = c.zMult;
    zAdd = c.zAdd;
}

float EngineVulkan::GetZCoef() const
{
    return 1.0f;
}

int EngineVulkan::Width() const
{
    return _w;
}

int EngineVulkan::Height() const
{
    return _h;
}

int EngineVulkan::Width2D() const
{
    return _w;
}

int EngineVulkan::Height2D() const
{
    return _h;
}

int EngineVulkan::AFrameTime() const
{
    return 0; // GL33 does the same
}

void EngineVulkan::SetGamma(float gamma)
{
    _gamma = gamma;
}

float EngineVulkan::GetGamma() const
{
    return _gamma;
}

// ── Draw path stubs (phase 1: 2D/UI, phase 2: world) ────────────────────────

void EngineVulkan::BeginMesh(TLVertexTable&, const render::LegacySpec&) {}
void EngineVulkan::EndMesh(TLVertexTable&) {}
void EngineVulkan::PrepareMesh(const render::LegacySpec&) {}
void EngineVulkan::PrepareTriangle(const PacLevelMem*, int) {}
void EngineVulkan::PrepareTriangle(const MipInfo&, int) {}
void EngineVulkan::DrawPolygon(const VertexIndex*, int) {}
void EngineVulkan::DrawPolygon(TLVertexTable&, const short*, int) {}
void EngineVulkan::DrawSection(const FaceArray&, Offset, Offset) {}
void EngineVulkan::DrawDecal(Vector3Par, float, float, float, PackedColor, const MipInfo&, int) {}
void EngineVulkan::Draw2D(const PacLevelMem*, PackedColor, float, float, float, float, float, float, float, float) {}
void EngineVulkan::Draw2D(const Draw2DPars&, const Rect2DAbs&, const Rect2DAbs&) {}
void EngineVulkan::DrawLine(int, int) {}
void EngineVulkan::DrawLine(const Line2DAbs&, PackedColor, PackedColor, const Rect2DAbs&) {}
void EngineVulkan::DrawPoly(const MipInfo&, const Vertex2DPixel*, int, const Rect2DPixel&, int) {}
void EngineVulkan::DrawPoly(const MipInfo&, const Vertex2DAbs*, int, const Rect2DAbs&, int) {}

AbstractTextBank* EngineVulkan::TextBank()
{
    return _bank; // TextBankDummy until phase 1 brings TextBankVulkan
}

void EngineVulkan::TextureDestroyed(Texture*) {}

void EngineVulkan::ResetForRemount() {}

Engine* CreateEngineVulkan(int w, int h, bool windowed, int bpp)
{
    EngineVulkan* engine = new EngineVulkan(w, h, windowed, bpp);
    if (!engine->IsUsable())
    {
        delete engine;
        return nullptr; // GameApplication falls back to Auto (gl33)
    }
    return engine;
}

} // namespace Poseidon
