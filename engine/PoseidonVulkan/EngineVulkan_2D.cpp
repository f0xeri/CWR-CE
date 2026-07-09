// 2D / screen-space producer entry points — line-for-line mirror of
// EngineGL33_2D.cpp + EngineGL33_DrawShared.cpp + the TLVertexTable part of
// EngineGL33_Mesh.cpp. All GL-free by construction: geometry building and
// clipping stay identical, the backend specifics live in AddVertices /
// QueuePrepareTriangle / FlushQueue (EngineVulkan_Queue.cpp).
#include "EngineVulkan.hpp"

#include <Poseidon/Graphics/Textures/TexturePreload.hpp>
#include <Poseidon/Graphics/Textures/TextureBank.hpp>
#include <Poseidon/Graphics/Rendering/Primitives/Clip2D.hpp>
#include <Poseidon/Graphics/Rendering/Primitives/Poly.hpp>
#include <Poseidon/World/Scene/Scene.hpp> // GPreloadedTextures (line texture)

namespace
{
inline int FracAlpha(float a)
{
    int ia = toInt(a);
    saturate(ia, 0, 255);
    return ia;
}
} // namespace

namespace Poseidon
{

void EngineVulkan::Draw2D(const Draw2DPars& pars, const Rect2DAbs& rect, const Rect2DAbs& clip)
{
    PoseidonAssert(pars.mip.IsOK());
    if (!pars.mip.IsOK())
        return;
    if (!IsUsable())
        return;

    // Pixel-corner addressing, same as GL33 (no D3D9 -0.5 shift).
    float xBeg = rect.x, xEnd = xBeg + rect.w;
    float yBeg = rect.y, yEnd = yBeg + rect.h;

    float uBeg = 0;
    float vBeg = 0;
    float uEnd = 1;
    float vEnd = 1;

    float xc = floatMax(clip.x, 0);
    float yc = floatMax(clip.y, 0);
    float xec = floatMin(clip.x + clip.w, _w);
    float yec = floatMin(clip.y + clip.h, _h);

    if (xBeg < xc)
    {
        uBeg = (xc - xBeg) / rect.w;
        xBeg = xc;
    }
    if (xEnd > xec)
    {
        uEnd = 1 - (xEnd - xec) / rect.w;
        xEnd = xec;
    }
    if (yBeg < yc)
    {
        vBeg = (yc - yBeg) / rect.h;
        yBeg = yc;
    }
    if (yEnd > yec)
    {
        vEnd = 1 - (yEnd - yec) / rect.h;
        yEnd = yec;
    }

    if (xBeg >= xEnd || yBeg >= yEnd)
        return;

    TLVertex pos[4];
    pos[0].rhw = 1;
    pos[0].color = pars.colorTL;
    pos[0].specular = PackedColor(0xff000000);
    pos[0].pos[2] = 0.5;

    pos[1].rhw = 1;
    pos[1].color = pars.colorTR;
    pos[1].specular = PackedColor(0xff000000);
    pos[1].pos[2] = 0.5;

    pos[2].rhw = 1;
    pos[2].color = pars.colorBR;
    pos[2].specular = PackedColor(0xff000000);
    pos[2].pos[2] = 0.5;

    pos[3].rhw = 1;
    pos[3].color = pars.colorBL;
    pos[3].specular = PackedColor(0xff000000);
    pos[3].pos[2] = 0.5;

    float uTL = pars.uTL + uBeg * (pars.uTR - pars.uTL) + vBeg * (pars.uBL - pars.uTL);
    float uTR = pars.uTL + uEnd * (pars.uTR - pars.uTL) + vBeg * (pars.uBL - pars.uTL);
    float uBL = pars.uTL + uBeg * (pars.uTR - pars.uTL) + vEnd * (pars.uBL - pars.uTL);
    float uBR = pars.uTL + uEnd * (pars.uTR - pars.uTL) + vEnd * (pars.uBL - pars.uTL);

    float vTL = pars.vTL + uBeg * (pars.vTR - pars.vTL) + vBeg * (pars.vBL - pars.vTL);
    float vTR = pars.vTL + uEnd * (pars.vTR - pars.vTL) + vBeg * (pars.vBL - pars.vTL);
    float vBL = pars.vTL + uBeg * (pars.vTR - pars.vTL) + vEnd * (pars.vBL - pars.vTL);
    float vBR = pars.vTL + uEnd * (pars.vTR - pars.vTL) + vEnd * (pars.vBL - pars.vTL);

    pos[0].pos[0] = xBeg, pos[0].pos[1] = yBeg;
    pos[1].pos[0] = xEnd, pos[1].pos[1] = yBeg;
    pos[2].pos[0] = xEnd, pos[2].pos[1] = yEnd;
    pos[3].pos[0] = xBeg, pos[3].pos[1] = yEnd;
    pos[0].t0.u = uTL, pos[0].t0.v = vTL;
    pos[1].t0.u = uTR, pos[1].t0.v = vTR;
    pos[2].t0.u = uBR, pos[2].t0.v = vBR;
    pos[3].t0.u = uBL, pos[3].t0.v = vBL;

    SwitchRenderMode(RM2DTris);
    BeginScreenPass();

    AddVertices(pos, 4); // note: may flush all queues

    QueuePrepareTriangle(pars.mip, pars.spec);

    Queue2DPoly(pos, 4);
}

void EngineVulkan::DrawLine(const Line2DAbs& line, PackedColor c0, PackedColor c1, const Rect2DAbs& clip)
{
    float x0 = line.beg.x;
    float y0 = line.beg.y;
    float x1 = line.end.x;
    float y1 = line.end.y;
    // use line texture
    Texture* tex = GPreloadedTextures.New(TextureLine);
    const MipInfo& mip = TextBank()->UseMipmap(tex, 1, 1);

    // convert line to poly;
    int specFlags = NoZBuf | IsAlpha | ClampU | ClampV | IsAlphaFog;
    float dx = x1 - x0;
    float dy = y1 - y0;
    float dSize2 = dx * dx + dy * dy;
    float invDSize = dSize2 > 0 ? InvSqrt(dSize2) : 1;

    float pdx = +dy * invDSize, pdy = -dx * invDSize;
    float w = 3.0f;
    x0 -= pdx * (w * 0.5);
    x1 -= pdx * (w * 0.5);
    y0 -= pdy * (w * 0.5);
    y1 -= pdy * (w * 0.5);
    float x0Side = x0 + pdx * w, y0Side = y0 + pdy * w;
    float x1Side = x1 + pdx * w, y1Side = y1 + pdy * w;

    Vertex2DAbs vertices[4];
    float off = 0;
    vertices[0].x = x0 - off;
    vertices[0].y = y0 - off;
    vertices[0].u = 0;
    vertices[0].v = 0.25;
    vertices[0].color = c0;

    vertices[1].x = x0Side - off;
    vertices[1].y = y0Side - off;
    vertices[1].u = 0;
    vertices[1].v = 1;
    vertices[1].color = c0;

    vertices[3].x = x1 - off;
    vertices[3].y = y1 - off;
    vertices[3].u = 0.1;
    vertices[3].v = 0.25;
    vertices[3].color = c1;

    vertices[2].x = x1Side - off;
    vertices[2].y = y1Side - off;
    vertices[2].u = 0.1;
    vertices[2].v = 1;
    vertices[2].color = c1;

    DrawPoly(mip, vertices, 4, clip, specFlags);
}

void EngineVulkan::DrawPoly(const MipInfo& mip, const Vertex2DPixel* vertices, int n, const Rect2DPixel& clipRect,
                            int specFlags)
{
    if (!IsUsable())
        return;

    const int maxN = 32;

    // reject poly if fully outside or invalid
    ClipFlags orClip = 0;
    ClipFlags andClip = ClipAll;
    for (int i = 0; i < n; i++)
    {
        const Vertex2DPixel& vs = vertices[i];
        float x = vs.x;
        float y = vs.y;
        ClipFlags clip = 0;
        if (x < clipRect.x)
            clip |= ClipLeft;
        else if (x > clipRect.x + clipRect.w)
            clip |= ClipRight;
        if (y < clipRect.y)
            clip |= ClipTop;
        else if (y > clipRect.y + clipRect.h)
            clip |= ClipBottom;
        orClip |= clip;
        andClip &= clip;
    }
    if (andClip)
        return;
    Vertex2DPixel clippedVertices1[maxN];
    Vertex2DPixel clippedVertices2[maxN];
    if (orClip)
    {
        Vertex2DPixel* free = clippedVertices1;
        Vertex2DPixel* used = clippedVertices2;
        for (int i = 0; i < n; i++)
            used[i] = vertices[i];
        if (orClip & ClipTop)
            n = Clip2D(clipRect, free, used, n, InsideTopPixel), swap(free, used);
        if (orClip & ClipBottom)
            n = Clip2D(clipRect, free, used, n, InsideBottomPixel), swap(free, used);
        if (orClip & ClipLeft)
            n = Clip2D(clipRect, free, used, n, InsideLeftPixel), swap(free, used);
        if (orClip & ClipRight)
            n = Clip2D(clipRect, free, used, n, InsideRightPixel), swap(free, used);
        if (n < 3)
            return;
        vertices = used;
    }

    if (n > maxN)
    {
        n = maxN;
        Fail("Poly: Too much vertices");
    }

    TLVertex gv[maxN];

    float x2d = Left2D();
    float y2d = Top2D();

    for (int i = 0; i < n; i++)
    {
        TLVertex* v = &gv[i];
        const Vertex2DPixel& vs = vertices[i];

        v->pos[0] = vs.x + x2d;
        v->pos[1] = vs.y + y2d;
        v->pos[2] = vs.z;
        v->rhw = vs.w;
        v->color = vs.color;
        v->specular = PackedColor(0xff000000);
        v->t0.u = vs.u;
        v->t0.v = vs.v;
    }

    SwitchRenderMode(RM2DTris);
    BeginScreenPass();
    AddVertices(gv, n); // note: may flush all queues
    QueuePrepareTriangle(mip, specFlags);
    Queue2DPoly(gv, n);
}

void EngineVulkan::DrawPoly(const MipInfo& mip, const Vertex2DAbs* vertices, int n, const Rect2DAbs& clipRect,
                            int specFlags)
{
    if (!IsUsable())
        return;

    const int maxN = 32;

    ClipFlags orClip = 0;
    ClipFlags andClip = ClipAll;
    for (int i = 0; i < n; i++)
    {
        const Vertex2DAbs& vs = vertices[i];
        float x = vs.x;
        float y = vs.y;
        ClipFlags clip = 0;
        if (x < clipRect.x)
            clip |= ClipLeft;
        else if (x > clipRect.x + clipRect.w)
            clip |= ClipRight;
        if (y < clipRect.y)
            clip |= ClipTop;
        else if (y > clipRect.y + clipRect.h)
            clip |= ClipBottom;
        orClip |= clip;
        andClip &= clip;
    }
    if (andClip)
        return;
    Vertex2DAbs clippedVertices1[maxN];
    Vertex2DAbs clippedVertices2[maxN];
    if (orClip)
    {
        Vertex2DAbs* free = clippedVertices1;
        Vertex2DAbs* used = clippedVertices2;
        for (int i = 0; i < n; i++)
            used[i] = vertices[i];
        if (orClip & ClipTop)
            n = Clip2D(clipRect, free, used, n, InsideTopAbs), swap(free, used);
        if (orClip & ClipBottom)
            n = Clip2D(clipRect, free, used, n, InsideBottomAbs), swap(free, used);
        if (orClip & ClipLeft)
            n = Clip2D(clipRect, free, used, n, InsideLeftAbs), swap(free, used);
        if (orClip & ClipRight)
            n = Clip2D(clipRect, free, used, n, InsideRightAbs), swap(free, used);
        if (n < 3)
            return;
        vertices = used;
    }

    if (n > maxN)
    {
        n = maxN;
        Fail("Poly: Too much vertices");
    }

    TLVertex gv[maxN];

    for (int i = 0; i < n; i++)
    {
        TLVertex* v = &gv[i];
        const Vertex2DAbs& vs = vertices[i];

        v->pos[0] = vs.x;
        v->pos[1] = vs.y;
        v->pos[2] = vs.z;
        v->rhw = vs.w;
        v->color = vs.color;
        v->specular = PackedColor(0xff000000);
        v->t0.u = vs.u;
        v->t0.v = vs.v;
    }

    SwitchRenderMode(RM2DTris);
    BeginScreenPass();
    AddVertices(gv, n); // note: may flush all queues
    QueuePrepareTriangle(mip, specFlags);
    Queue2DPoly(gv, n);
}

void EngineVulkan::DrawLine(int beg, int end)
{
    if (!_mesh)
        return;
    const TLVertex& v0 = _mesh->GetVertex(beg);
    const TLVertex& v1 = _mesh->GetVertex(end);

    float x0 = v0.pos.X();
    float y0 = v0.pos.Y();
    float x1 = v1.pos.X();
    float y1 = v1.pos.Y();

    float z0 = v0.pos.Z();
    float z1 = v1.pos.Z();
    float w0 = v0.rhw;
    float w1 = v1.rhw;

    // use line texture
    Texture* tex = GPreloadedTextures.New(TextureLine);
    const MipInfo& mip = TextBank()->UseMipmap(tex, 1, 1);

    // convert line to poly;
    int specFlags = NoZWrite | IsAlpha | ClampU | ClampV | IsAlphaFog;
    float dx = x1 - x0;
    float dy = y1 - y0;
    float dSize2 = dx * dx + dy * dy;
    float invDSize = dSize2 > 0 ? InvSqrt(dSize2) : 1;

    float dSize = dSize2 * invDSize;

    float pdx = +dy * invDSize, pdy = -dx * invDSize;
    float w = 3.0f;
    x0 -= pdx * (w * 0.5);
    x1 -= pdx * (w * 0.5);
    y0 -= pdy * (w * 0.5);
    y1 -= pdy * (w * 0.5);
    float x0Side = x0 + pdx * w, y0Side = y0 + pdy * w;
    float x1Side = x1 + pdx * w, y1Side = y1 + pdy * w;

    Vertex2DAbs vertices[4];
    float off = 0.0f;
    vertices[0].x = x0 - off;
    vertices[0].y = y0 - off;
    vertices[0].z = z0;
    vertices[0].w = w0;
    vertices[0].u = 0;
    vertices[0].v = 0.25;
    vertices[0].color = v0.color;

    vertices[1].x = x0Side - off;
    vertices[1].y = y0Side - off;
    vertices[1].z = z0;
    vertices[1].w = w0;
    vertices[1].u = 0;
    vertices[1].v = 1;
    vertices[1].color = v0.color;

    vertices[2].x = x1Side - off;
    vertices[2].y = y1Side - off;
    vertices[2].z = z1;
    vertices[2].w = w1;
    vertices[2].u = dSize;
    vertices[2].v = 1;
    vertices[2].color = v1.color;

    vertices[3].x = x1 - off;
    vertices[3].y = y1 - off;
    vertices[3].z = z1;
    vertices[3].w = w1;
    vertices[3].u = dSize;
    vertices[3].v = 0.25;
    vertices[3].color = v1.color;

    Rect2DAbs clip(0, 0, _w, _h);

    DrawPoly(mip, vertices, 4, clip, specFlags);
}

// ── Shared decal / polygon / point paths (mirror of EngineGL33_DrawShared) ──

void EngineVulkan::DrawDecal(Vector3Par screen, float rhw, float sizeX, float sizeY, PackedColor color,
                             const MipInfo& mip, int specFlags)
{
    float vx = screen.X();
    float vy = screen.Y();
    float z = screen.Z();

    float oow = rhw;

    // perform simple clipping
    float xBeg = vx - sizeX;
    float xEnd = vx + sizeX;
    float yBeg = vy - sizeY;
    float yEnd = vy + sizeY;
    float uBeg = 0;
    float vBeg = 0;
    float uEnd = 1;
    float vEnd = 1;

    if (xBeg < 0)
    {
        uBeg = -xBeg / (2 * sizeX);
        xBeg = 0;
    }
    if (xEnd > _w)
    {
        uEnd = 1 - (xEnd - _w) / (2 * sizeX);
        xEnd = _w;
    }
    if (yBeg < 0)
    {
        vBeg = -yBeg / (2 * sizeY);
        yBeg = 0;
    }
    if (yEnd > _h)
    {
        vEnd = 1 - (yEnd - _h) / (2 * sizeY);
        yEnd = _h;
    }

    if (xBeg >= xEnd || yBeg >= yEnd)
        return;

    TLVertex v[4];

    v[0].pos[0] = xBeg;
    v[0].pos[1] = yBeg;
    v[0].pos[2] = z;
    v[0].t0.u = uBeg;
    v[0].t0.v = vBeg;
    v[1].pos[0] = xEnd;
    v[1].pos[1] = yBeg;
    v[1].pos[2] = z;
    v[1].t0.u = uEnd;
    v[1].t0.v = vBeg;
    v[2].pos[0] = xEnd;
    v[2].pos[1] = yEnd;
    v[2].pos[2] = z;
    v[2].t0.u = uEnd;
    v[2].t0.v = vEnd;
    v[3].pos[0] = xBeg;
    v[3].pos[1] = yEnd;
    v[3].pos[2] = z;
    v[3].t0.u = uBeg;
    v[3].t0.v = vEnd;

    if (specFlags & IsAlphaFog)
    {
        v[0].color = color;
        v[0].specular = PackedColor(0xff000000);
    }
    else
    {
        // use fog with z
        v[0].specular = PackedColor(0xff000000 - (color & 0xff000000));
        v[0].color = PackedColor(color | 0xff000000);
    }

    for (int i = 0; i < 4; i++)
    {
        v[i].rhw = oow;
        v[i].pos[2] = z;
        v[i].color = v[0].color;
        v[i].specular = v[0].specular;
    }

    SwitchRenderMode(RMTris);
    BeginScreenPass();

    AddVertices(v, 4);

    QueuePrepareTriangle(mip, specFlags);
    static const VertexIndex indices[4] = {0, 1, 2, 3};
    QueueFan(indices, 4);
}

void EngineVulkan::DrawPolygon(const VertexIndex* ii, int n)
{
    QueueFan(ii, n);
}

void EngineVulkan::DrawSection(const FaceArray& face, Offset beg, Offset end)
{
    if (!IsUsable())
        return;
    for (Offset i = beg; i < end; face.Next(i))
    {
        const Poly& f = face[i];
        QueueFan(f.GetVertexList(), f.N());
    }
}

void EngineVulkan::DrawPoints(const TLVertex* vs, int nVertex)
{
    if (!IsUsable())
        return;
    for (int i = 0; i < nVertex; i++)
    {
        const TLVertex& v = vs[i];
        PackedColor color = v.color;
        if (color.A8() < 8)
            continue; // do not draw stars that are not visible

        int xI = toIntFloor(v.pos[0]);
        int yI = toIntFloor(v.pos[1]);
        float xFrac = v.pos[0] - xI;
        float yFrac = v.pos[1] - yI;
        float ixFrac = 1 - xFrac;
        float iyFrac = 1 - yFrac;

        if (xI < 0 || xI + 2 > _w || yI < 0 || yI + 2 > _h)
            continue; // all four corners must be on screen

        float a = color.A8();

        float aTL = FracAlpha(ixFrac * iyFrac * a);
        float aTR = FracAlpha(xFrac * iyFrac * a);
        float aBR = FracAlpha(xFrac * yFrac * a);
        float aBL = FracAlpha(ixFrac * yFrac * a);
        TLVertex quad[4];
        quad[0] = v; // TL
        quad[0].pos[0] = xI + 0.5f;
        quad[0].pos[1] = yI + 0.5f;
        quad[0].color = PackedColorRGB(color, aTL);
        // No fog on screen-space points (vFogTC = specular.a), see GL33.
        quad[0].specular = PackedColor(0xff000000);

        quad[1] = quad[0]; // TR
        quad[1].pos[0] = xI + 2.5f;
        quad[1].color = PackedColorRGB(color, aTR);

        quad[2] = quad[1]; // BR
        quad[2].pos[1] = yI + 2.5f;
        quad[2].color = PackedColorRGB(color, aBR);

        quad[3] = quad[2]; // BL
        quad[3].pos[0] = quad[0].pos[0];
        quad[3].color = PackedColorRGB(color, aBL);

        static const VertexIndex indices[4] = {0, 1, 2, 3};

        AddVertices(quad, 4);
        QueueFan(indices, 4);
    }
}

void EngineVulkan::DrawPoints(int beg, int end)
{
    if (!IsUsable() || !_mesh)
        return;

    for (int i = beg; i < end; i++)
    {
        if (_mesh->Clip(i) & ClipAll)
            continue;

        const TLVertex& v = _mesh->GetVertex(i);
        PackedColor color = v.color;
        if (color.A8() < 8)
            continue;

        int xI = toIntFloor(v.pos[0]);
        int yI = toIntFloor(v.pos[1]);
        float xFrac = v.pos[0] - xI;
        float yFrac = v.pos[1] - yI;
        float ixFrac = 1 - xFrac;
        float iyFrac = 1 - yFrac;

        float a = color.A8();

        TLVertex quad[4];
        quad[0] = v; // TL
        quad[0].pos[0] = xI + 0.5f;
        quad[0].pos[1] = yI + 0.5f;
        quad[0].color = PackedColorRGB(color, FracAlpha(ixFrac * iyFrac * a));
        quad[0].specular = PackedColor(0xff000000);

        quad[1] = quad[0]; // TR
        quad[1].pos[0] = xI + 2.5f;
        quad[1].color = PackedColorRGB(color, FracAlpha(xFrac * iyFrac * a));

        quad[2] = quad[1]; // BR
        quad[2].pos[1] = yI + 2.5f;
        quad[2].color = PackedColorRGB(color, FracAlpha(xFrac * yFrac * a));

        quad[3] = quad[2]; // BL
        quad[3].pos[0] = quad[0].pos[0];
        quad[3].color = PackedColorRGB(color, FracAlpha(ixFrac * yFrac * a));

        static const VertexIndex indices[4] = {0, 1, 2, 3};

        AddVertices(quad, 4);
        QueueFan(indices, 4);
    }
}

// ── TLVertexTable path (mirror of EngineGL33_Mesh.cpp's screen part) ────────

void EngineVulkan::PrepareMesh(const render::LegacySpec& /*spec*/)
{
    BeginScreenPass();
}

void EngineVulkan::BeginMesh(TLVertexTable& mesh, const render::LegacySpec& /*spec*/)
{
    BeginScreenPass();
    _mesh = &mesh;

    AddVertices(mesh.VertexData(), mesh.NVertex());
}

void EngineVulkan::EndMesh(TLVertexTable& /*mesh*/)
{
    _mesh = nullptr;
}

} // namespace Poseidon
