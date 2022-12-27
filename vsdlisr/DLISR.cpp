#include "DLISR.h"
#include <cuda_runtime.h>
#include <mutex>

using namespace std::literals;

std::mutex g_pages_mutex;

static const VSFrame *VS_CC dlisr_get_frame(int n, int activationReason,
                                            void *instanceData,
                                            [[maybe_unused]] void **frameData,
                                            VSFrameContext *frameCtx,
                                            VSCore *core, const VSAPI *vsapi) {
  auto d{static_cast<PluginData *>(instanceData)};

  if (activationReason == arInitial) {
    vsapi->requestFrameFilter(n, d->node, frameCtx);
    return nullptr;
  }

  if (activationReason != arAllFramesReady)
    return nullptr;

  auto src{vsapi->getFrameFilter(n, d->node, frameCtx)};
  decltype(src) fr[]{nullptr, nullptr, nullptr};
  constexpr int pl[]{0, 1, 2};

  auto dst{vsapi->newVideoFrame2(&d->src_vi->format, d->dst_vi.width,
                                 d->dst_vi.height, fr, pl, src, core)};

  uint8_t *srcp[]{(uint8_t *)vsapi->getReadPtr(src, 0),
                  (uint8_t *)vsapi->getReadPtr(src, 1),
                  (uint8_t *)vsapi->getReadPtr(src, 2)};

  uint8_t *dstp[]{(uint8_t *)vsapi->getWritePtr(dst, 0),
                  (uint8_t *)vsapi->getWritePtr(dst, 1),
                  (uint8_t *)vsapi->getWritePtr(dst, 2)};

  uint8_t *interleaved_data{d->interleaved.data()};

  std::lock_guard<std::mutex> guard(g_pages_mutex);

  for (int px{0}; px < d->src_vi->width * d->src_vi->height; px++) {
    interleaved_data[px * 3 + 0] = srcp[0][px];
    interleaved_data[px * 3 + 1] = srcp[1][px];
    interleaved_data[px * 3 + 2] = srcp[2][px];
  }

  if (cudaMemcpy(d->in_image_dev_ptr, d->interleaved.data(), d->in_image_size,
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    throw "Error copying input RGB image to CUDA buffer";
  }

  // Synchronize the device.
  cudaDeviceSynchronize();

  // Execute the feature.
  NVSDK_NGX_CUDA_EvaluateFeature(d->DUHandle, d->params, nullptr);

  // Synchronize once more.
  cudaDeviceSynchronize();

  // Copy output image from GPU device memory
  if (cudaMemcpy(d->out_image.data(), d->out_image_dev_ptr, d->out_image_size,
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    throw "Error copying output image from CUDA buffer";
  }

  uint8_t *out_image_data{d->out_image.data()};

  for (int px{0}; px < d->dst_vi.width * d->dst_vi.height; px++) {
    dstp[0][px] = out_image_data[px * 3 + 0];
    dstp[1][px] = out_image_data[px * 3 + 1];
    dstp[2][px] = out_image_data[px * 3 + 2];
  }

  vsapi->freeFrame(src);
  return dst;
}

static void VS_CC dlisr_free(void *instanceData, [[maybe_unused]] VSCore *core,
                             const VSAPI *vsapi) {
  auto d{static_cast<PluginData *>(instanceData)};

  cudaFree(d->in_image_dev_ptr);
  cudaFree(d->out_image_dev_ptr);

  vsapi->freeNode(d->node);

  delete d;
}

static void VS_CC dlisr_create(const VSMap *in, VSMap *out,
                               [[maybe_unused]] void *userData, VSCore *core,
                               const VSAPI *vsapi) {
  auto d{std::make_unique<PluginData>()};

  try {
    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);

    int err{9};
    d->upres_factor = vsapi->mapGetIntSaturated(in, "rfactor", 0, &err);
    if (err)
      d->upres_factor = 2;

    d->src_vi = vsapi->getVideoInfo(d->node);

    d->dst_vi = *d->src_vi;
    d->dst_vi.width = d->src_vi->width * d->upres_factor;
    d->dst_vi.height = d->src_vi->height * d->upres_factor;

    if (!vsh::isConstantVideoFormat(d->src_vi) ||
        d->src_vi->format.sampleType != stInteger ||
        d->src_vi->format.bitsPerSample != 8 ||
        d->src_vi->format.colorFamily != cfRGB) {
      throw "only constant format 8 bit integer RGB supported";
    }

    // size is also size in bytes
    d->in_image_size = d->src_vi->width * d->src_vi->height * 3;
    d->out_image_size = d->dst_vi.width * d->dst_vi.height * 3;

    if (cudaMalloc(&d->in_image_dev_ptr, d->in_image_size) != cudaSuccess) {
      throw "Error allocating input image CUDA buffer";
    }

    if (cudaMalloc(&d->out_image_dev_ptr, d->out_image_size) != cudaSuccess) {
      throw "Error allocating output image CUDA buffer";
    }

    // Initialize NGX.
    NVSDK_NGX_Result rslt = NVSDK_NGX_Result_Success;
    rslt = NVSDK_NGX_CUDA_Init(0x0, L"./", NVSDK_NGX_Version_API);
    if (rslt != NVSDK_NGX_Result_Success) {
      throw "Error Initializing NGX. ";
    }

    // Get the parameter block.
    NVSDK_NGX_CUDA_GetParameters(&d->params);

    // Verify feature is supported
    int Supported = 0;
    d->params->Get(NVSDK_NGX_Parameter_ImageSuperResolution_Available,
                   &Supported);
    if (!Supported) {
      throw "NVSDK_NGX_Feature_ImageSuperResolution Unavailable on this System";
    }

    // Set the default hyperparameters for inferrence.
    d->params->Set(NVSDK_NGX_Parameter_Width, d->src_vi->width);
    d->params->Set(NVSDK_NGX_Parameter_Height, d->src_vi->height);
    d->params->Set(NVSDK_NGX_Parameter_Scale, d->upres_factor);

    d->interleaved.resize(d->in_image_size);
    d->out_image.resize(d->out_image_size);

    // Pass the pointers to the GPU allocations to thes
    // parameter block along with the format and size.
    d->params->Set(NVSDK_NGX_Parameter_Color_SizeInBytes, d->in_image_size);
    d->params->Set(NVSDK_NGX_Parameter_Color_Format,
                   NVSDK_NGX_Buffer_Format_RGB8UI);
    d->params->Set(NVSDK_NGX_Parameter_Color, d->in_image_dev_ptr);
    d->params->Set(NVSDK_NGX_Parameter_Output_SizeInBytes, d->out_image_size);
    d->params->Set(NVSDK_NGX_Parameter_Output_Format,
                   NVSDK_NGX_Buffer_Format_RGB8UI);
    d->params->Set(NVSDK_NGX_Parameter_Output, d->out_image_dev_ptr);

    // Get the scratch buffer size and create the scratch allocation.
    // (if required)
    size_t byteSize{0u};
    void *scratchBuffer{nullptr};
    rslt = NVSDK_NGX_CUDA_GetScratchBufferSize(
        NVSDK_NGX_Feature_ImageSuperResolution, d->params, &byteSize);
    if (rslt != NVSDK_NGX_Result_Success) {
      throw "Error Getting NGX Scratch Buffer Size. ";
    }
    cudaMalloc(&scratchBuffer, byteSize > 0u ? byteSize : 1u);
    // cudaMalloc, unlike malloc, fails on 0 size allocations

    // Update the parameter block with the scratch space metadata.:
    d->params->Set(NVSDK_NGX_Parameter_Scratch, scratchBuffer);
    d->params->Set(NVSDK_NGX_Parameter_Scratch_SizeInBytes, (uint32_t)byteSize);

    // Create the feature
    NVSDK_NGX_CUDA_CreateFeature(NVSDK_NGX_Feature_ImageSuperResolution,
                                 d->params, &d->DUHandle);

  } catch (const char *error) {
    vsapi->mapSetError(out, ("DLISR: "s + error).c_str());
    vsapi->freeNode(d->node);
    return;
  }

  VSFilterDependency deps[]{{d->node, rpStrictSpatial}};
  vsapi->createVideoFilter(out, "DLISR", &d->dst_vi, dlisr_get_frame,
                           dlisr_free, fmParallel, deps, 1, d.get(), core);
  d.release();
}

//////////////////////////////////////////
// Init

VS_EXTERNAL_API(void)
VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
  vspapi->configPlugin("moe.grass.vsdlisr", "vsdlisr", "vsdlisr",
                       VS_MAKE_VERSION(1, 0), VAPOURSYNTH_API_VERSION, 0,
                       plugin);
  vspapi->registerFunction("DLISR",
                           "clip:vnode;"
                           "rfactor:int:opt;",
                           "clip:vnode;", dlisr_create, nullptr, plugin);
}
