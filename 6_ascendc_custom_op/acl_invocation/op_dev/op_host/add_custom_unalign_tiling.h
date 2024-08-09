/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef ADD_CUSTOM_UNALIGN_TILING_H
#define ADD_CUSTOM_UNALIGN_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TilingDataUnalign)
  TILING_DATA_FIELD_DEF(uint32_t, formerNum);
  TILING_DATA_FIELD_DEF(uint32_t, tailNum);
  TILING_DATA_FIELD_DEF(uint32_t, formerLength);
  TILING_DATA_FIELD_DEF(uint32_t, tailLength);
  TILING_DATA_FIELD_DEF(uint32_t, alignNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AddCustomUnalign, TilingDataUnalign)
}
#endif // ADD_CUSTOM_UNALIGN_TILING_H
