#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2023 Huawei Technologies Co., Ltd
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
torch.npu.config.allow_internal_format = False


class TestCustomAdd(TestCase):
    def test_add_custom(self):
        length = [8, 2048]
        x = torch.rand(length, device='cpu', dtype=torch.float16)
        y = torch.rand(length, device='cpu', dtype=torch.float16)
        print(x, '\n', y)

        prof_path = "./prof_total"
        with torch.npu.profile(prof_path) as prof:
            torch.npu.synchronize()
            output = torch_npu.npu_add_custom(x.npu(), y.npu()).cpu()
            torch.npu.synchronize()

        print(output)
        self.assertRtolEqual(output, x + y)


if __name__ == "__main__":
    run_tests()
