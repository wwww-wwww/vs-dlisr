#pragma once

#include "Vapoursynth/VSHelper4.h"
#include "Vapoursynth/VapourSynth4.h"

#include "nvsdk_ngx.h"

#include <vector>

struct PluginData final {
  VSNode *node;
  const VSVideoInfo *src_vi;
  VSVideoInfo dst_vi;

  int upres_factor;

  NVSDK_NGX_Handle *DUHandle{nullptr};
  NVSDK_NGX_Parameter *params{nullptr};

  void *in_image_dev_ptr;
  void *out_image_dev_ptr;

  size_t in_image_size;
  size_t out_image_size;

  std::vector<uint8_t> interleaved;
  std::vector<uint8_t> out_image;
};
