// Path-B mesh buffers — mirror of EngineGL33_VertexBuffer.cpp. Static shapes
// live in device-local vertex+index buffers filled through the texture-upload
// staging path (the copy lands in the frame's upload CB, which is submitted
// before the frame's draws). Dynamic (animated) shapes re-copy their vertices
// into the per-frame host-visible vertex ring on every Update, replacing
// GL33's map-with-INVALIDATE orphaning.
#include "EngineVulkan.hpp"

#include <Poseidon/Foundation/Logging/Logging.hpp>
#include <Poseidon/Graphics/Rendering/Primitives/Poly.hpp>
#include <Poseidon/Graphics/Rendering/Shape/Shape.hpp>

#include <climits>

namespace Poseidon
{

// Index range per mesh section.
struct VBSectionInfoVulkan
{
    int beg, end;
    int begVertex, endVertex;
};

class VertexBufferVulkan : public VertexBuffer
{
    friend class EngineVulkan;

  private:
    EngineVulkan* _engine = nullptr;
    // Static path: device-local vertex buffer.
    vk::Buffer _vb;
    VmaAllocation _vbAlloc = nullptr;
    // Dynamic path: this draw's window inside the frame vertex ring.
    vk::Buffer _dynBuffer;
    vk::DeviceSize _dynOffset = 0;
    // Indices are always static (topology never animates).
    vk::Buffer _ib;
    VmaAllocation _ibAlloc = nullptr;

    bool _dynamic = false;
    int _vertexCount = 0;
    int _indexCount = 0;
    std::vector<VBSectionInfoVulkan> _sections;

  public:
    explicit VertexBufferVulkan(EngineVulkan* engine) : _engine(engine) {}
    ~VertexBufferVulkan() override;

    bool Init(const Shape& src, VBType type);
    void Update(const Shape& src, bool dynamic) override;

  private:
    static void FillVertices(SVertexVulkan* dst, const Shape& src);
    bool UploadStatic(const Shape& src);
    bool UpdateDynamic(const Shape& src);
};

VertexBufferVulkan::~VertexBufferVulkan()
{
    // The GPU may still consume the buffers this frame — route through the
    // fence-gated deferred destroy like texture surfaces.
    if (_engine)
    {
        _engine->DeferDestroyBuffer(_vb, _vbAlloc);
        _engine->DeferDestroyBuffer(_ib, _ibAlloc);
    }
    _vb = nullptr;
    _ib = nullptr;
}

void VertexBufferVulkan::FillVertices(SVertexVulkan* dst, const Shape& src)
{
    const UVPair* uv = &src.UV(0);
    const Vector3* pos = &src.Pos(0);
    const Vector3* norm = &src.Norm(0);
    for (int i = src.NVertex(); --i >= 0;)
    {
        dst->pos = Vector3P(pos->X(), pos->Y(), pos->Z());
        // Normals are negated (matches the D3D-era convention GL33 mirrors)
        dst->norm = Vector3P(-norm->X(), -norm->Y(), -norm->Z());
        pos++;
        norm++;
        dst->t0 = *uv;
        uv++;
        dst++;
    }
}

// Stages the current shape vertices into the device-local vertex buffer.
bool VertexBufferVulkan::UploadStatic(const Shape& src)
{
    const vk::DeviceSize bytes = vk::DeviceSize(_vertexCount) * sizeof(SVertexVulkan);
    EngineVulkan::UploadTicket ticket = _engine->BeginTextureUpload();
    vk::Buffer staging;
    vk::DeviceSize stagingOffset = 0;
    uint8_t* dst = _engine->AllocStaging(ticket, bytes, staging, stagingOffset);
    if (!dst)
    {
        _engine->EndTextureUpload(ticket);
        return false;
    }
    FillVertices(reinterpret_cast<SVertexVulkan*>(dst), src);
    ticket.cmd.copyBuffer(staging, _vb, vk::BufferCopy(stagingOffset, 0, bytes));
    // Make the copy visible to vertex fetch. The upload CB runs before the
    // frame's draws, so a submission-order barrier is sufficient.
    vk::BufferMemoryBarrier2 barrier(vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite,
                                     vk::PipelineStageFlagBits2::eVertexInput,
                                     vk::AccessFlagBits2::eVertexAttributeRead, vk::QueueFamilyIgnored,
                                     vk::QueueFamilyIgnored, _vb, 0, bytes);
    ticket.cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, barrier, nullptr));
    _engine->EndTextureUpload(ticket);
    return true;
}

// Copies the animated vertices into the frame vertex ring; the section draw
// binds the ring at the returned offset. Called from BeginMeshTL on every
// draw of a dynamic shape (GL33 re-maps the VBO just as often).
bool VertexBufferVulkan::UpdateDynamic(const Shape& src)
{
    const vk::DeviceSize bytes = vk::DeviceSize(_vertexCount) * sizeof(SVertexVulkan);
    void* dst = _engine->AllocDynamicMeshVertices(bytes, _dynBuffer, _dynOffset);
    if (!dst)
        return false;
    FillVertices(static_cast<SVertexVulkan*>(dst), src);
    return true;
}

bool VertexBufferVulkan::Init(const Shape& src, VBType type)
{
    if (src.NVertex() <= 0)
    {
        LOG_DEBUG(Graphics, "VK: Empty vertices.");
        return false;
    }

    _dynamic = (type == VBDynamic || type == VBSmallDiscardable);
    _vertexCount = src.NVertex();

    VulkanContext& ctx = _engine->Context();

    if (!_dynamic)
    {
        const VkBufferCreateInfo vbInfo =
            vk::BufferCreateInfo({}, vk::DeviceSize(_vertexCount) * sizeof(SVertexVulkan),
                                 vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                                 vk::SharingMode::eExclusive);
        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
        VkBuffer raw = VK_NULL_HANDLE;
        if (vmaCreateBuffer(ctx.allocator, &vbInfo, &allocInfo, &raw, &_vbAlloc, nullptr) != VK_SUCCESS)
        {
            LOG_ERROR(Graphics, "VK: mesh vertex buffer allocation failed ({} verts)", _vertexCount);
            return false;
        }
        _vb = raw;
        if (!UploadStatic(src))
            return false;
    }
    // Dynamic shapes don't upload here: BeginMeshTL refreshes their ring
    // window before every draw, and outside an open frame there is no ring.

    // Count total indices (fan triangulation: N-gon -> N-2 triangles).
    int indices = 0;
    for (Offset o = src.BeginFaces(); o < src.EndFaces(); src.NextFace(o))
    {
        const Poly& poly = src.Face(o);
        PoseidonAssert(poly.N() >= 3);
        indices += (poly.N() - 2) * 3;
    }
    _indexCount = indices;

    if (indices > 0)
    {
        const vk::DeviceSize ibBytes = vk::DeviceSize(indices) * sizeof(VertexIndex);
        const VkBufferCreateInfo ibInfo =
            vk::BufferCreateInfo({}, ibBytes,
                                 vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                                 vk::SharingMode::eExclusive);
        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
        VkBuffer raw = VK_NULL_HANDLE;
        if (vmaCreateBuffer(ctx.allocator, &ibInfo, &allocInfo, &raw, &_ibAlloc, nullptr) != VK_SUCCESS)
        {
            LOG_ERROR(Graphics, "VK: mesh index buffer allocation failed ({} indices)", indices);
            return false;
        }
        _ib = raw;

        EngineVulkan::UploadTicket ticket = _engine->BeginTextureUpload();
        vk::Buffer staging;
        vk::DeviceSize stagingOffset = 0;
        uint8_t* dst = _engine->AllocStaging(ticket, ibBytes, staging, stagingOffset);
        if (!dst)
        {
            _engine->EndTextureUpload(ticket);
            return false;
        }
        VertexIndex* iData = reinterpret_cast<VertexIndex*>(dst);
        for (Offset o = src.BeginFaces(); o < src.EndFaces(); src.NextFace(o))
        {
            const Poly& poly = src.Face(o);
            for (int i = 2; i < poly.N(); i++)
            {
                *iData++ = poly.GetVertex(0);
                *iData++ = poly.GetVertex(i - 1);
                *iData++ = poly.GetVertex(i);
            }
        }
        ticket.cmd.copyBuffer(staging, _ib, vk::BufferCopy(stagingOffset, 0, ibBytes));
        vk::BufferMemoryBarrier2 barrier(vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite,
                                         vk::PipelineStageFlagBits2::eIndexInput, vk::AccessFlagBits2::eIndexRead,
                                         vk::QueueFamilyIgnored, vk::QueueFamilyIgnored, _ib, 0, ibBytes);
        ticket.cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, barrier, nullptr));
        _engine->EndTextureUpload(ticket);

        // Per-section index ranges (GL33 mirror).
        _sections.resize(src.NSections());
        int start = 0;
        for (int i = 0; i < src.NSections(); i++)
        {
            const ShapeSection& sec = src.GetSection(i);
            int size = 0;
            int minV = INT_MAX;
            int maxV = 0;
            for (Offset o = sec.beg; o < sec.end; src.NextFace(o))
            {
                const Poly& face = src.Face(o);
                PoseidonAssert(face.N() >= 3);
                size += (face.N() - 2) * 3;
                for (int vv = 0; vv < face.N(); vv++)
                {
                    int vi = face.GetVertex(vv);
                    saturateMin(minV, vi);
                    saturateMax(maxV, vi);
                }
            }
            _sections[i].beg = start;
            _sections[i].end = start + size;
            _sections[i].begVertex = minV;
            _sections[i].endVertex = maxV + 1;
            start += size;
        }
    }

    return true;
}

void VertexBufferVulkan::Update(const Shape& src, bool dynamic)
{
    if (_dynamic)
    {
        // Dynamic shapes refresh their ring window every draw: the previous
        // window may belong to a retired frame slot.
        UpdateDynamic(src);
        bufferDirty = false;
        return;
    }
    if (dynamic || bufferDirty)
    {
        UploadStatic(src);
        bufferDirty = false;
    }
}

VertexBuffer* EngineVulkan::CreateVertexBuffer(const Shape& src, VBType type)
{
    auto* buf = new VertexBufferVulkan(this);
    if (buf->Init(src, type))
        return buf;
    delete buf;
    return nullptr;
}

// Dynamic mesh vertices live in the frame's vertex ring after the soup
// window; allocations are valid for the current frame only.
void* EngineVulkan::AllocDynamicMeshVertices(vk::DeviceSize bytes, vk::Buffer& outBuffer, vk::DeviceSize& outOffset)
{
    if (!_frameOpen)
        return nullptr;
    VulkanBuffer& ring = _dynMeshBuffer[_frameIndex];
    void* p = ring.Allocate(bytes, 16, outOffset);
    if (!p)
    {
        if (!_soupOverflowLogged)
        {
            LOG_ERROR(Graphics, "VK: dynamic mesh ring exhausted — draws dropped for the rest of the frame");
            _soupOverflowLogged = true;
        }
        return nullptr;
    }
    outBuffer = ring.buffer;
    return p;
}

void EngineVulkan::DrawSectionTL(const Shape& sMesh, int beg, int end)
{
    auto* buf = static_cast<VertexBufferVulkan*>(sMesh.GetVertexBuffer());
    if (!buf || buf->_sections.empty() || !_frameOpen || !_pipelineBound)
        return;

    PoseidonAssert(end > beg);
    PoseidonAssert(end <= (int)buf->_sections.size());

    const VBSectionInfoVulkan& siBeg = buf->_sections[beg];
    const VBSectionInfoVulkan& siEnd = buf->_sections[end - 1];

    const int indexCount = siEnd.end - siBeg.beg;
    if (indexCount <= 0)
        return;

    const vk::Buffer vb = buf->_dynamic ? buf->_dynBuffer : buf->_vb;
    if (!vb || !buf->_ib)
        return;

    vk::CommandBuffer cmd = _frames[_frameIndex].cmd;

    // Bind (or reuse) the material/pass descriptor set, then push the
    // per-draw world matrix — the Vulkan analogue of GL33's 64-byte
    // world-subrange UBO update.
    if (!WriteConstantsAndBindDescriptors(cmd))
        return;
    cmd.pushConstants(_pipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, 64, &_currentDrawItem.worldMatrix);

    cmd.bindVertexBuffers(0, vb, buf->_dynamic ? buf->_dynOffset : vk::DeviceSize(0));
    cmd.bindIndexBuffer(buf->_ib, 0, vk::IndexType::eUint16);
    cmd.drawIndexed(indexCount, 1, siBeg.beg, 0, 0);
    ++gPerfDrawCalls;
}

} // namespace Poseidon
