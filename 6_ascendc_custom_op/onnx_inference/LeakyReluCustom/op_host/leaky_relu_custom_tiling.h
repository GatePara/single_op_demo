#ifndef LEAKYRELU_CUSTOM_TILING_H
#define LEAKYRELU_CUSTOM_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(float, negativeSlope);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LeakyReluCustom, TilingData)
}
#endif // LEAKYRELU_CUSTOM_TILING_H