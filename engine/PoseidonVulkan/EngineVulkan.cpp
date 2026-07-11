#include "EngineVulkan.hpp"
#include "TextureVulkan.hpp"

#include <Poseidon/Core/Application.hpp>
#include <Poseidon/Core/Config/EngineConfig.hpp>
#include <Poseidon/Graphics/Dummy/TextBankDummy.hpp>
#include <Poseidon/Foundation/Logging/Logging.hpp>
#include <Poseidon/Graphics/Core/ZBiasMath.hpp>
#include <Poseidon/Graphics/Shared/WindowPlacement.hpp>
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#ifdef _WIN32
#include <windows.h> // hardware gamma ramp (SetDeviceGammaRamp)
#endif

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

    _sdlWindow = SDL_CreateWindow("Poseidon [Vulkan 1.3]", placement.width, placement.height, flags);
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

    if (!_swapchain.Create(_vk, _sdlWindow) || !CreateFrameResources() || !InitPipelineResources())
    {
        LOG_ERROR(Graphics, "VK: swapchain/frame/pipeline setup failed — engine unusable");
        DestroyPipelineResources();
        DestroyFrameResources();
        _swapchain.Destroy(_vk);
        _vk.Shutdown();
        SDL_DestroyWindow(_sdlWindow);
        _sdlWindow = nullptr;
        return;
    }
    ResetPSConstantDefaults();

    // GPU is alive: swap the bootstrap TextBankDummy for the real bank.
    delete _bank;
    _bank = new TextBankVulkan(this);

    int cw = 0, ch = 0;
    SDL_GetWindowSizeInPixels(_sdlWindow, &cw, &ch);
    _w = cw;
    _h = ch;
    LOG_INFO(Graphics, "VK: surface resolved to {}x{} {}", _w, _h, _windowed ? "windowed" : "fullscreen");

    _eventWindow.Attach(_sdlWindow, _w, _h);

    InitDebugOverlay();

    LoadConfig();
}

EngineVulkan::~EngineVulkan()
{
    ClearFontCache();
    // Bank teardown pushes surface images into the deferred-delete queues;
    // they are flushed below once the device is idle.
    delete _bank;
    _bank = nullptr;
    _eventWindow.Detach();
    if (_vk.IsValid())
    {
        _vk.device.waitIdle(); // in-flight submits may still reference frame resources
        ShutdownDebugOverlay(); // ImGui's Vulkan objects must die before the device
        FlushAllDeferredDestroys();
        DestroyPipelineResources();
        DestroyFrameResources();
        _swapchain.Destroy(_vk);
    }
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

// Frame cycle (InitDraw/FinishDraw/NextFrame/Clear/OnWindowResized) lives in
// EngineVulkan_Frame.cpp.

void EngineVulkan::Pause() {}
void EngineVulkan::Restore() {}
void EngineVulkan::DrawPicture555(unsigned short*) {}

// ── Lighting / atmosphere hooks (FogColorChanged lives in _Shaders.cpp) ─────

void EngineVulkan::LightChanged(const Color&, const Color&) {}
void EngineVulkan::NightEffectChanged(float) {}

// ── Display modes ───────────────────────────────────────────────────────────
// Unlike GL33 there is no context reset dance: every path just moves the SDL
// window and marks the swapchain dirty; the next InitDraw rebuilds it at the
// surface's new size (RecreateSwapchain also resizes the depth target).

bool EngineVulkan::ApplyExclusiveFullscreen(int w, int h, int refresh)
{
    SDL_DisplayID display = SDL_GetDisplayForWindow(_sdlWindow);
    if (!display)
        display = SDL_GetPrimaryDisplay();
    SDL_DisplayMode mode;
    if (SDL_GetClosestFullscreenDisplayMode(display, w, h, (float)refresh, false, &mode))
    {
        if (!SDL_SetWindowFullscreenMode(_sdlWindow, &mode))
            LOG_WARN(Graphics, "VK: SDL_SetWindowFullscreenMode failed for {}x{}@{}: {}", w, h, refresh,
                     SDL_GetError());
    }
    else
    {
        LOG_WARN(Graphics, "VK: no fullscreen mode close to {}x{}@{} — using desktop mode", w, h, refresh);
        SDL_SetWindowFullscreenMode(_sdlWindow, nullptr);
    }
    return SDL_SetWindowFullscreen(_sdlWindow, true);
}

bool EngineVulkan::SwitchRes(int w, int h, int bpp)
{
    if (!_sdlWindow)
        return false;
    _pixelSize = bpp;

    if (_windowed)
    {
        SDL_SetWindowSize(_sdlWindow, w, h);
        _windowedRestoreW = w;
        _windowedRestoreH = h;
    }
    else if (_windowMode == WindowMode::Fullscreen)
    {
        if (!ApplyExclusiveFullscreen(w, h, _refreshRate))
            LOG_WARN(Graphics, "VK: SDL_SetWindowFullscreen failed: {}", SDL_GetError());
    }
    // Borderless always covers the monitor — a resolution request is a no-op.

    SDL_GetWindowSizeInPixels(_sdlWindow, &_w, &_h);
    _swapchainDirty = true;
    LOG_INFO(Graphics, "VK: SwitchRes {}x{} {}bpp -> surface {}x{}", w, h, bpp, _w, _h);
    return true;
}

bool EngineVulkan::SwitchRefreshRate(int refresh)
{
    if (refresh == 0)
        return false;
    if (_refreshRate == refresh)
        return true;
    _refreshRate = refresh;
    if (_windowed || _windowMode != WindowMode::Fullscreen)
        return true;
    if (ApplyExclusiveFullscreen(_w, _h, refresh))
        _swapchainDirty = true;
    return true;
}

bool EngineVulkan::SetWindowMode(WindowMode mode)
{
    if (!_sdlWindow)
        return false;
    if (_windowed && mode != WindowMode::Windowed)
        SDL_GetWindowSize(_sdlWindow, &_windowedRestoreW, &_windowedRestoreH);
    _windowMode = mode;

    switch (mode)
    {
        case WindowMode::Windowed:
            SDL_SetWindowFullscreen(_sdlWindow, false);
            SDL_SetWindowBordered(_sdlWindow, true);
            SDL_SetWindowResizable(_sdlWindow, true);
            if (_windowedRestoreW > 0 && _windowedRestoreH > 0)
                SDL_SetWindowSize(_sdlWindow, _windowedRestoreW, _windowedRestoreH);
            SDL_SetWindowPosition(_sdlWindow, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
            _windowed = true;
            break;
        case WindowMode::Borderless:
            // Desktop-fullscreen: nullptr mode keeps the desktop resolution.
            SDL_SetWindowFullscreenMode(_sdlWindow, nullptr);
            if (!SDL_SetWindowFullscreen(_sdlWindow, true))
                LOG_WARN(Graphics, "VK: borderless switch failed: {}", SDL_GetError());
            _windowed = false;
            break;
        case WindowMode::Fullscreen:
            if (!ApplyExclusiveFullscreen(_w, _h, _refreshRate))
                LOG_WARN(Graphics, "VK: exclusive fullscreen switch failed: {}", SDL_GetError());
            _windowed = false;
            break;
    }

    SDL_GetWindowSizeInPixels(_sdlWindow, &_w, &_h);
    _swapchainDirty = true;
    LOG_INFO(Graphics, "VK: SetWindowMode {} -> surface {}x{}",
             mode == WindowMode::Fullscreen   ? "fullscreen"
             : mode == WindowMode::Borderless ? "borderless"
                                              : "windowed",
             _w, _h);
    return true;
}

bool EngineVulkan::SetSwapInterval(int interval)
{
    if (_swapInterval == interval)
        return true;
    _swapInterval = interval;
    _swapchainDirty = true; // present mode is baked into the swapchain
    LOG_INFO(Graphics, "VK: swap interval {} — swapchain rebuild scheduled", interval);
    return true;
}

bool EngineVulkan::GetDesktopDisplayMode(int& w, int& h, int& refresh) const
{
    SDL_DisplayID display = _sdlWindow ? SDL_GetDisplayForWindow(_sdlWindow) : SDL_GetPrimaryDisplay();
    const SDL_DisplayMode* dm = SDL_GetDesktopDisplayMode(display ? display : SDL_GetPrimaryDisplay());
    if (!dm)
        return false;
    w = dm->w;
    h = dm->h;
    refresh = (int)(dm->refresh_rate + 0.5f);
    return true;
}

bool EngineVulkan::GetRequestedFullscreenMode(int& w, int& h, int& refresh) const
{
    if (_windowed)
        return false;
    w = _w;
    h = _h;
    refresh = _refreshRate;
    return true;
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
    saturate(gamma, 1e-3f, 1e3f);
    _gamma = gamma;

    // Hardware gamma ramp on the window's device context — identical to
    // GL33's DoSetGamma; the ramp is display-level, not graphics-API-level.
#ifdef _WIN32
    if (!_sdlWindow)
        return;
    SDL_PropertiesID props = SDL_GetWindowProperties(_sdlWindow);
    HWND hwnd = (HWND)SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WIN32_HWND_POINTER, NULL);
    if (!hwnd)
        return;
    HDC hdc = GetDC(hwnd);
    if (!hdc)
        return;

    WORD ramp[3][256];
    const float eGamma = 1.0f / _gamma;
    ramp[0][0] = ramp[1][0] = ramp[2][0] = 0;
    for (int i = 1; i < 256; i++)
    {
        const float x = i * (1.0f / 255.0f);
        const float fx = powf(x, eGamma);
        int ifx = static_cast<int>(fx * 65535.0f);
        if (ifx < 0)
            ifx = 0;
        if (ifx > 65535)
            ifx = 65535;
        ramp[0][i] = ramp[1][i] = ramp[2][i] = static_cast<WORD>(ifx);
    }
    SetDeviceGammaRamp(hdc, ramp);
    ReleaseDC(hwnd, hdc);
    LOG_DEBUG(Graphics, "VK: set gamma {:.3f}", _gamma);
#endif
}

float EngineVulkan::GetGamma() const
{
    return _gamma;
}

// ── Remaining draw-path stubs (2D/soup paths live in EngineVulkan_2D.cpp) ───

void EngineVulkan::PrepareTriangle(const PacLevelMem*, int) {}
void EngineVulkan::DrawPolygon(TLVertexTable&, const short*, int) {}
void EngineVulkan::Draw2D(const PacLevelMem*, PackedColor, float, float, float, float, float, float, float, float) {}

AbstractTextBank* EngineVulkan::TextBank()
{
    return _bank; // TextBankVulkan (bootstrap TextBankDummy until the GPU is up)
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
    // Construct first so SDL video / window are initialised, then hide the OS
    // cursor unconditionally — the game draws its own cursor sprite (mirrors
    // CreateEngineGL33; SDL_HideCursor before the window exists is a no-op on
    // some SDL3 backends, so the order matters).
    SDL_HideCursor();
    return engine;
}

} // namespace Poseidon
