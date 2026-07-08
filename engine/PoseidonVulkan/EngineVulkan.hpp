#pragma once

#include <Poseidon/Graphics/Core/Engine.hpp>
#include <PoseidonGL33/SDLEventWindow.hpp>
#include "VulkanContext.hpp"

struct SDL_Window;

namespace Poseidon
{
class TextBankVulkan;

class EngineVulkan : public Engine
{
  private:
    // TODO(vk-phase1): real TextBankVulkan; until then a TextBankDummy keeps
    // the producer layer alive (Scene::Init etc. dereference TextBank()
    // unconditionally).
    AbstractTextBank* _bank;

  public:
    EngineVulkan(int width, int height, bool windowed, int bpp);
    ~EngineVulkan() override;

    bool IsUsable() const { return _sdlWindow != nullptr && _vk.IsValid(); }

    RString GetDebugName() const override;
    RString GetRendererName() const override;
    using Engine::InitDraw;
    void InitDraw();
    void FinishDraw() override;
    void Pause() override;
    void Restore() override;
    void DrawPicture555(unsigned short*);
    void FogColorChanged(const Color&) override;
    void LightChanged(const Color&, const Color&);
    void NightEffectChanged(float);

    bool SwitchRes(int w, int h, int bpp) override;
    bool SwitchRefreshRate(int refresh) override;
    bool SetWindowMode(WindowMode mode) override;

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
};

Engine* CreateEngineVulkan(int w, int h, bool windowed, int bpp);

} // namespace Poseidon
