"""
* @file generate_data.py
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
"""
import numpy as np

a = np.random.randint(100, size=(1, 999,)).astype(np.float16)
b = np.random.randint(100, size=(1, 999,)).astype(np.float16)

a.tofile('input_0.bin')
b.tofile('input_1.bin')
