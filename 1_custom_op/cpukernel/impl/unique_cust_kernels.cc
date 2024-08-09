/* Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "unique_cust_kernels.h"

#include "cpu_tensor.h"
#include "cpu_tensor_shape.h"
#include "cpu_types.h"
#include "cust_cpu_utils.h"

namespace {
const char* UNIQUE_CUST = "UniqueCust";
const uint32_t kFirstInputIndex = 0;
const uint32_t kFirstOutputIndex = 0;
const uint32_t kSecondOutputIndex = 1;
const uint32_t SUCCESS = 0;
const uint32_t PARAM_INVAILD = 1;

template <typename Tin, typename Tidx>
uint32_t UniqueTask(aicpu::Tensor *x, aicpu::Tensor *y, aicpu::Tensor *idx,
                    int64_t N) {
  Tin *a = reinterpret_cast<Tin *>(x->GetData());
  if (a == nullptr) {
    return PARAM_INVAILD;
  }

  Tin *out = reinterpret_cast<Tin *>(y->GetData());
  if (out == nullptr) {
    return PARAM_INVAILD;
  }

  Tidx *idx_vec = reinterpret_cast<Tidx *>(idx->GetData());
  if (idx_vec == nullptr) {
    return PARAM_INVAILD;
  }

  std::unordered_map<Tin, Tidx> uniq;
  uniq.reserve(2 * N);
  for (Tidx i = 0, j = 0; i < N; ++i) {
    auto it = uniq.emplace(a[i], j);
    idx_vec[i] = it.first->second;
    if (it.second) {
      ++j;
    }
  }
  for (const auto &it : uniq) {
    out[it.second] = it.first;
  }

  // update outputshape
  auto y_shape = y->GetTensorShape();
  if (y_shape == nullptr) {
    return PARAM_INVAILD;
  }

  if (y_shape->GetUnknownRank()) {
    std::vector<int64_t> y_shape_values = y_shape->GetDimSizes();
    if (y_shape_values.size() == 0) {
      y_shape_values.push_back(uniq.size());
    } else {
      y_shape_values[0] = uniq.size();
    }

    y_shape->SetDimSizes(y_shape_values);
  }
  return SUCCESS;
}
}

namespace aicpu {
static std::map<int32_t, std::map<int32_t, 
  std::function<uint32_t(aicpu::Tensor *, aicpu::Tensor *, aicpu::Tensor *, int64_t)>>> unique_calls = {
    {DataType::DT_UINT8, {{DataType::DT_INT32, UniqueTask<uint8_t, int32_t>},
                          {DataType::DT_INT64, UniqueTask<uint8_t, int64_t>}}},
    {DataType::DT_UINT16, {{DataType::DT_INT32, UniqueTask<uint16_t, int32_t>},
                           {DataType::DT_INT64, UniqueTask<uint16_t, int64_t>}}},
    {DataType::DT_INT8, {{DataType::DT_INT32, UniqueTask<int8_t, int32_t>},
                         {DataType::DT_INT64, UniqueTask<int8_t, int64_t>}}},
    {DataType::DT_INT16, {{DataType::DT_INT32, UniqueTask<int16_t, int32_t>},
                          {DataType::DT_INT64, UniqueTask<int16_t, int64_t>}}},
    {DataType::DT_INT32, {{DataType::DT_INT32, UniqueTask<int32_t, int32_t>},
                          {DataType::DT_INT64, UniqueTask<int32_t, int64_t>}}},
    {DataType::DT_INT64, {{DataType::DT_INT32, UniqueTask<int64_t, int32_t>},
                          {DataType::DT_INT64, UniqueTask<int64_t, int64_t>}}},
    {DataType::DT_FLOAT, {{DataType::DT_INT32, UniqueTask<float, int32_t>},
                          {DataType::DT_INT64, UniqueTask<float, int64_t>}}},
    {DataType::DT_DOUBLE, {{DataType::DT_INT32, UniqueTask<double, int32_t>},
                           {DataType::DT_INT64, UniqueTask<double, int64_t>}}}
  };

uint32_t UniqueCpuKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_LOG_DEBUG(ctx, "Start Cust UniqueCpuKernel Compute");
  Tensor *param_tensor = ctx.Input(kFirstInputIndex);
  if (param_tensor == nullptr) {
    return PARAM_INVAILD;
  }
  auto param_shape = param_tensor->GetTensorShape();
  if (param_shape == nullptr) {
    return PARAM_INVAILD;
  }

  DataType param_type = param_tensor->GetDataType();
  int64_t p_size = 1;
  for (int i = 0; i < param_shape->GetDims(); ++i) {
    p_size *= param_shape->GetDimSize(i);
  }

  AttrValue *out_idx_attr = ctx.GetAttr("out_idx");
  auto out_idx_type = (out_idx_attr == nullptr) ? DataType::DT_INT32 :
                      (out_idx_attr->GetDataType());
  CUST_KERNEL_LOG_DEBUG(ctx, "Cust UniqueCpuKernel Compute, p_size is %ld, out_idx = %d.",
                        p_size, out_idx_type);

  const auto &func_map = unique_calls.find(param_type);
  if (func_map != unique_calls.end()) {
    const auto &func = func_map->second.find(out_idx_type);
    if (func != func_map->second.end()) {
      return (func->second)(param_tensor, ctx.Output(kFirstOutputIndex),
                            ctx.Output(kSecondOutputIndex), p_size);
    }
  }

  CUST_KERNEL_LOG_ERROR(ctx, "UniqueCust op kernel input dtype[%d], output dtype[%d] not support.",
                        param_type, out_idx_type);
  return PARAM_INVAILD;
}

REGISTER_CPU_KERNEL(UNIQUE_CUST, UniqueCpuKernel);
}  // namespace aicpu
