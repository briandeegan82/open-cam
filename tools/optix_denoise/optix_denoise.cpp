// OptiX denoiser for planar float32 layers (tools/denoise_optix.py drives it; docs/DENOISING.md).
//
//   optix_denoise <aov|hdr> W H N_LAYERS LAYER_CH in.f32 albedo.f32 normal.f32 out.f32
//
// in.f32: the beauty (H x W x 3) followed by N_LAYERS layers of H x W x LAYER_CH, float32, rows top to bottom.
// LAYER_CH 1 is rejected by the OptiX 7.7 AOV model (FLOAT1 layers: invalid value); use 3. albedo/normal: H x W x 3 (normal in camera space; the non-temporal models use x and y).
// aov: one OPTIX_DENOISER_MODEL_KIND_AOV call; the filter is predicted from the beauty and the guides and applied to
//      every layer, so all layers are filtered alike.
// hdr: OPTIX_DENOISER_MODEL_KIND_HDR on every 3-channel layer separately (LAYER_CH must be 3), like OIDN.
// out.f32: same layout as in.f32 (the beauty denoised, then the layers).

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define CUDA_CHECK(x)                                                                         \
    do {                                                                                      \
        cudaError_t e = (x);                                                                  \
        if (e != cudaSuccess) {                                                               \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e)); \
            std::exit(2);                                                                     \
        }                                                                                     \
    } while (0)
#define OPTIX_CHECK(x)                                                                        \
    do {                                                                                      \
        OptixResult r = (x);                                                                  \
        if (r != OPTIX_SUCCESS) {                                                             \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, optixGetErrorString(r)); \
            std::exit(2);                                                                     \
        }                                                                                     \
    } while (0)

static std::vector<float> read_f32(const char *path, size_t n) {
    std::vector<float> v(n);
    std::ifstream f(path, std::ios::binary);
    if (!f.read(reinterpret_cast<char *>(v.data()), n * sizeof(float))) {
        std::fprintf(stderr, "cannot read %zu floats from %s\n", n, path);
        std::exit(2);
    }
    return v;
}

static CUdeviceptr upload(const float *p, size_t n) {
    void *d;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, p, n * sizeof(float), cudaMemcpyHostToDevice));
    return reinterpret_cast<CUdeviceptr>(d);
}

static CUdeviceptr alloc(size_t bytes) {
    void *d;
    CUDA_CHECK(cudaMalloc(&d, bytes ? bytes : 1));
    return reinterpret_cast<CUdeviceptr>(d);
}

static OptixImage2D image(CUdeviceptr data, unsigned w, unsigned h, unsigned ch) {
    OptixImage2D im = {};
    im.data = data;
    im.width = w;
    im.height = h;
    im.rowStrideInBytes = w * ch * sizeof(float);
    im.pixelStrideInBytes = ch * sizeof(float);
    im.format = ch == 1 ? OPTIX_PIXEL_FORMAT_FLOAT1 : OPTIX_PIXEL_FORMAT_FLOAT3;
    return im;
}

int main(int argc, char **argv) {
    if (argc != 10) {
        std::fprintf(stderr, "usage: %s <aov|hdr> W H N_LAYERS LAYER_CH in.f32 albedo.f32 normal.f32 out.f32\n", argv[0]);
        return 1;
    }
    const std::string mode = argv[1];
    const unsigned w = std::atoi(argv[2]), h = std::atoi(argv[3]), nl = std::atoi(argv[4]), lc = std::atoi(argv[5]);
    if ((mode != "aov" && mode != "hdr") || (lc != 1 && lc != 3) || (mode == "hdr" && lc != 3)) {
        std::fprintf(stderr, "bad mode/layer channels (hdr needs 3)\n");
        return 1;
    }
    const size_t px = size_t(w) * h, total = px * (3 + size_t(nl) * lc);
    std::vector<float> in = read_f32(argv[6], total);
    std::vector<float> alb = read_f32(argv[7], px * 3), nrm = read_f32(argv[8], px * 3);

    CUDA_CHECK(cudaFree(nullptr));
    OPTIX_CHECK(optixInit());
    OptixDeviceContext ctx;
    OPTIX_CHECK(optixDeviceContextCreate(nullptr, nullptr, &ctx));

    OptixDenoiserOptions opt = {};
    opt.guideAlbedo = 1;
    opt.guideNormal = 1;
    OptixDenoiser dn;
    OPTIX_CHECK(optixDenoiserCreate(ctx, mode == "aov" ? OPTIX_DENOISER_MODEL_KIND_AOV : OPTIX_DENOISER_MODEL_KIND_HDR,
                                    &opt, &dn));
    OptixDenoiserSizes sz;
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(dn, w, h, &sz));
    size_t scratchBytes = sz.withoutOverlapScratchSizeInBytes;
    scratchBytes = std::max(scratchBytes, std::max(sz.computeIntensitySizeInBytes, sz.computeAverageColorSizeInBytes));
    CUdeviceptr state = alloc(sz.stateSizeInBytes), scratch = alloc(scratchBytes);
    OPTIX_CHECK(optixDenoiserSetup(dn, nullptr, w, h, state, sz.stateSizeInBytes, scratch, scratchBytes));

    CUdeviceptr dIn = upload(in.data(), total), dOut = alloc(total * sizeof(float));
    OptixDenoiserGuideLayer guide = {};
    guide.albedo = image(upload(alb.data(), px * 3), w, h, 3);
    guide.normal = image(upload(nrm.data(), px * 3), w, h, 3);

    // layer i (0 = beauty) starts at float offset off(i)
    auto off = [&](unsigned i) { return i == 0 ? size_t(0) : px * (3 + size_t(i - 1) * lc); };
    auto layer = [&](unsigned i) {
        OptixDenoiserLayer l = {};
        unsigned ch = i == 0 ? 3 : lc;
        l.input = image(dIn + off(i) * sizeof(float), w, h, ch);
        l.output = image(dOut + off(i) * sizeof(float), w, h, ch);
        return l;
    };

    CUdeviceptr intensity = alloc(sizeof(float)), avgColor = alloc(3 * sizeof(float));
    OptixDenoiserParams params = {};
    params.hdrIntensity = intensity;
    if (mode == "aov") {
        std::vector<OptixDenoiserLayer> layers;
        for (unsigned i = 0; i <= nl; ++i) layers.push_back(layer(i));
        OPTIX_CHECK(optixDenoiserComputeIntensity(dn, nullptr, &layers[0].input, intensity, scratch, scratchBytes));
        OPTIX_CHECK(optixDenoiserComputeAverageColor(dn, nullptr, &layers[0].input, avgColor, scratch, scratchBytes));
        params.hdrAverageColor = avgColor;
        OPTIX_CHECK(optixDenoiserInvoke(dn, nullptr, &params, state, sz.stateSizeInBytes, &guide, layers.data(),
                                        unsigned(layers.size()), 0, 0, scratch, scratchBytes));
    } else {
        for (unsigned i = 0; i <= nl; ++i) {
            OptixDenoiserLayer l = layer(i);
            OPTIX_CHECK(optixDenoiserComputeIntensity(dn, nullptr, &l.input, intensity, scratch, scratchBytes));
            OPTIX_CHECK(optixDenoiserInvoke(dn, nullptr, &params, state, sz.stateSizeInBytes, &guide, &l, 1, 0, 0,
                                            scratch, scratchBytes));
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> out(total);
    CUDA_CHECK(cudaMemcpy(out.data(), reinterpret_cast<void *>(dOut), total * sizeof(float), cudaMemcpyDeviceToHost));
    std::ofstream f(argv[9], std::ios::binary);
    f.write(reinterpret_cast<const char *>(out.data()), total * sizeof(float));
    optixDenoiserDestroy(dn);
    optixDeviceContextDestroy(ctx);
    return f ? 0 : 2;
}
