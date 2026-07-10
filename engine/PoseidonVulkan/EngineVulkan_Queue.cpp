// Path-A soup queues — structural mirror of EngineGL33_Queue.cpp. Vertices
// and indices go straight into the frame's persistently mapped buffers (no
// CPU mirror / deferred upload: the memcpy IS the upload), a flush records
// bind + draw into the frame's command buffer. Window-advance replaces GL33's
// buffer orphaning so 16-bit indices stay valid (doc 4.7, path A).
#include "EngineVulkan.hpp"

#include <Poseidon/Foundation/Logging/Logging.hpp>
#include <Poseidon/Graphics/Core/FanDecompose.hpp>

namespace Poseidon
{

int QueueVulkan::Allocate(Texture* tex, int level, int spec, int minI, int maxI, int tip)
{
    int index = -1;
    if (tip >= minI && tip < maxI && _triUsed[tip])
    {
        TriQueueVulkan& triq = _tri[tip];
        if (tex == triq._texture && spec == triq._special)
            index = tip;
    }

    int free = -1;
    if (index < 0)
    {
        for (int i = minI; i < maxI; i++)
        {
            if (_triUsed[i])
            {
                TriQueueVulkan& triq = _tri[i];
                if (tex != triq._texture || spec != triq._special)
                    continue;
                index = i;
            }
            else if (free < 0)
            {
                free = i;
            }
        }
    }
    _usedCounter++;
    if (index >= 0)
    {
        TriQueueVulkan& triq = _tri[index];
        saturateMin(triq._level, level);
        triq._lastUsed = _usedCounter;
        return index;
    }
    if (free >= 0)
    {
        TriQueueVulkan& triq = _tri[free];
        triq._special = spec;
        triq._texture = tex;
        triq._level = level;
        triq._passId = SpecToPassId(spec);
        triq._lastUsed = _usedCounter;
        PoseidonAssert(triq._triangleQueue.Size() == 0);
        triq._triangleQueue.Resize(0);
        _triUsed[free] = true;
    }
    return free;
}

void QueueVulkan::Free(int i)
{
    PoseidonAssert(_tri[i]._triangleQueue.Size() == 0);
    PoseidonAssert(_triUsed[i]);
    _triUsed[i] = false;
}

WORD* EngineVulkan::QueueAdd(QueueVulkan& queue, int n)
{
    if (_instCount > 1)
        _instImpure = true; // soup-queue geometry can't be instanced — run must fall back

    PoseidonAssert(queue._actTri >= 0);
    PoseidonAssert(queue._triUsed[queue._actTri]);
    TriQueueVulkan& triq = queue._tri[queue._actTri];
    if (triq._triangleQueue.Size() + n > QueueVulkan::TriQueueSize)
        FlushQueue(queue, queue._actTri);

    int index = triq._triangleQueue.Size();
    triq._triangleQueue.Resize(index + n);
    return triq._triangleQueue.Data() + index;
}

void EngineVulkan::QueueFan(const VertexIndex* ii, int n)
{
    const int addN = render::geom::FanTriangleIndexCount(n);
    if (addN == 0)
        return;

    WORD* tgt = QueueAdd(_queueNo, addN);
    if (!tgt || !ii)
        return;

    const int offset = _queueNo._meshBase;
    PoseidonAssert(offset >= 0);

    render::geom::FanToTriangles(ii, n, offset, tgt);
}

void EngineVulkan::Queue2DPoly(const TLVertex* /*v0*/, int n)
{
    int addN = (n - 2) * 3;
    PoseidonAssert(_queueNo._actTri >= 0);
    PoseidonAssert(_queueNo._triUsed[_queueNo._actTri]);
    WORD* tgt = QueueAdd(_queueNo, addN);

    int offset = _queueNo._meshBase;
    PoseidonAssert(offset >= 0);

    for (int i = 2; i < n; i++)
    {
        *tgt++ = 0 + offset;
        *tgt++ = i - 1 + offset;
        *tgt++ = i + offset;
    }
}

void EngineVulkan::FlushQueue(QueueVulkan& queue, int index)
{
    TriQueueVulkan& triq = queue._tri[index];
    const int n = triq._triangleQueue.Size();
    if (n <= 0)
        return;
    if (!_frameOpen)
    {
        // No open command buffer (minimized / acquire failed) — drop.
        triq._triangleQueue.Clear();
        return;
    }

    if (index == QueueVulkan::MaxTriQueues - 1)
        FlushAllQueues(queue, index);

    ApplyPassState(triq._texture, triq._level, render::SplitLegacy(triq._special), triq._passId);
    if (!_pipelineBound)
    {
        triq._triangleQueue.Clear();
        return;
    }

    vk::CommandBuffer cmd = _frames[_frameIndex].cmd;

    // Index window handling (GL33's orphan-on-overflow analogue).
    if (queue._indexBufferUsed + n > QueueVulkan::IndexBufferLength)
    {
        _indexWindowBase += queue._indexBufferUsed;
        queue._indexBufferUsed = 0;
    }
    VulkanBuffer& ib = _indexBuffer[_frameIndex];
    const size_t ibWritePos = (size_t)(_indexWindowBase + queue._indexBufferUsed);
    if ((ibWritePos + n) * sizeof(WORD) > ib.capacity)
    {
        if (!_soupOverflowLogged)
        {
            LOG_ERROR(Graphics, "VK: index ring exhausted — draws dropped for the rest of the frame");
            _soupOverflowLogged = true;
        }
        triq._triangleQueue.Clear();
        return;
    }
    memcpy(ib.mapped + ibWritePos * sizeof(WORD), triq._triangleQueue.Data(), n * sizeof(WORD));
    const int firstIndex = queue._indexBufferUsed;
    queue._indexBufferUsed += n;

    // Bind both windows; indices are relative to the vertex window base and
    // firstIndex to the index window base, exactly like GL33's buffer offsets.
    cmd.bindVertexBuffers(0, _vertexBuffer[_frameIndex].buffer,
                          vk::DeviceSize(_vertexWindowBase * sizeof(TLVertex)));
    cmd.bindIndexBuffer(ib.buffer, vk::DeviceSize(_indexWindowBase * sizeof(WORD)), vk::IndexType::eUint16);

    if (WriteConstantsAndBindDescriptors(cmd))
    {
        cmd.drawIndexed(n, 1, firstIndex, 0, 0);
        ++gPerfDrawCalls;
    }

    triq._triangleQueue.Clear();
}

void EngineVulkan::FlushAndFreeQueue(QueueVulkan& queue, int index)
{
    FlushQueue(queue, index);
    FreeQueue(queue, index);
}

int EngineVulkan::AllocateQueue(QueueVulkan& queue, Texture* tex, int level, int spec)
{
    const bool alpha = (tex != nullptr && tex->IsAlpha()) || !_enableReorder;
    int minI = 0;
    int maxI = QueueVulkan::MaxTriQueues - 1;
    if (alpha)
    {
        minI = QueueVulkan::MaxTriQueues - 1;
        maxI = QueueVulkan::MaxTriQueues;
        FlushAllQueues(queue, QueueVulkan::MaxTriQueues - 1);
    }

    int index = queue.Allocate(tex, level, spec, minI, maxI, queue._actTri);
    if (index >= 0)
    {
        PoseidonAssert(queue._triUsed[index]);
        return index;
    }
    // Free LRU queue
    int minUsed = INT_MAX;
    for (int i = minI; i < maxI; i++)
    {
        int used = queue._tri[i]._lastUsed;
        if (used < minUsed)
        {
            minUsed = used;
            index = i;
        }
    }
    if (index < 0)
        index = 0;
    FlushAndFreeQueue(queue, index);
    index = queue.Allocate(tex, level, spec, minI, maxI, index);
    PoseidonAssert(index >= 0);
    PoseidonAssert(queue._triUsed[index]);
    return index;
}

void EngineVulkan::FreeQueue(QueueVulkan& queue, int index)
{
    PoseidonAssert(!queue._tri[index]._triangleQueue.Size());
    queue.Free(index);
}

void EngineVulkan::FreeAllQueues(QueueVulkan& queue)
{
    for (int i = 0; i < QueueVulkan::MaxTriQueues; i++)
    {
        if (queue._triUsed[i])
        {
            queue._tri[i]._triangleQueue.Clear();
            FreeQueue(queue, i);
        }
    }
}

void EngineVulkan::FlushAndFreeAllQueues(QueueVulkan& queue, bool nonEmptyOnly)
{
    for (int i = 0; i < QueueVulkan::MaxTriQueues; i++)
    {
        if (queue._triUsed[i] && (!nonEmptyOnly || queue._tri[i]._triangleQueue.Size() > 0))
            FlushAndFreeQueue(queue, i);
    }
}

void EngineVulkan::FlushAllQueues(QueueVulkan& queue, int skip)
{
    for (int i = 0; i < QueueVulkan::MaxTriQueues; i++)
    {
        if (i != skip && queue._triUsed[i])
            FlushQueue(queue, i);
    }
}

void EngineVulkan::DoSwitchRenderMode(RenderMode mode)
{
    FlushAndFreeAllQueues(_queueNo);
    _renderMode = mode;
}

void EngineVulkan::QueuePrepareTriangle(const MipInfo& absMip, int specFlags)
{
    _queueNo._actTri = AllocateQueue(_queueNo, absMip._texture, absMip._level, specFlags);
    PoseidonAssert(_queueNo._triUsed[_queueNo._actTri]);
}

void EngineVulkan::PrepareTriangle(const MipInfo& absMip, int specFlags0)
{
    SwitchRenderMode(RMTris);
    BeginScreenPass();
    _queueNo._actTri = AllocateQueue(_queueNo, absMip._texture, absMip._level, specFlags0);
    PoseidonAssert(_queueNo._triUsed[_queueNo._actTri]);
    _prepSpec = specFlags0;
}

void EngineVulkan::BeginScreenPass()
{
    if (!IsIn3DPass())
        return;
    FlushAndFreeAllQueues(_queueNo);
    _activePassId = PassId::ScreenSpace;

    // Reset the IsColored tint so a leftover mesh value can't dim the HUD.
    static const float white[4] = {1, 1, 1, 1};
    UploadPSConstant(3 /*SlotConstColor*/, white);
    UploadVSScreenConstants();
}

void EngineVulkan::AddVertices(const TLVertex* v, int n)
{
    if (n <= 0 || !_frameOpen)
        return;
    if (n > QueueVulkan::MeshBufferLength)
    {
        LOG_ERROR(Graphics, "Needed {} vertices, {} available", n, (int)QueueVulkan::MeshBufferLength);
        return;
    }

    // Window overflow: flush everything referencing the current window, then
    // slide the window forward (GL33 orphans + restarts at 0 instead).
    if (_queueNo._vertexBufferUsed + n > QueueVulkan::MeshBufferLength)
    {
        FlushAndFreeAllQueues(_queueNo);
        _vertexWindowBase += _queueNo._vertexBufferUsed;
        _queueNo._vertexBufferUsed = 0;
    }

    VulkanBuffer& vb = _vertexBuffer[_frameIndex];
    const size_t writePos = (size_t)(_vertexWindowBase + _queueNo._vertexBufferUsed);
    if ((writePos + n) * sizeof(TLVertex) > vb.capacity)
    {
        if (!_soupOverflowLogged)
        {
            LOG_ERROR(Graphics, "VK: vertex ring exhausted — draws dropped for the rest of the frame");
            _soupOverflowLogged = true;
        }
        return;
    }
    memcpy(vb.mapped + writePos * sizeof(TLVertex), v, sizeof(TLVertex) * n);
    _queueNo._meshBase = _queueNo._vertexBufferUsed;
    _queueNo._meshSize = n;
    _queueNo._vertexBufferUsed += n;
}

void EngineVulkan::EnableReorderQueues(bool enableReorder)
{
    if (_enableReorder == enableReorder)
        return;
    _enableReorder = enableReorder;
    if (!_enableReorder)
        FlushQueues();
}

void EngineVulkan::FlushQueues()
{
    FlushAndFreeAllQueues(_queueNo);
}

// Per-poly shadow brackets: only flush ordering matters until the stencil
// path lands in phase 3 (see GL33's comments in EngineGL33_Draw.cpp).
void EngineVulkan::BeginShadowPass()
{
    FlushAndFreeAllQueues(_queueNo, /*nonEmptyOnly*/ true);
}

void EngineVulkan::EndShadowPass()
{
    FlushAndFreeAllQueues(_queueNo, /*nonEmptyOnly*/ true);
}

} // namespace Poseidon
