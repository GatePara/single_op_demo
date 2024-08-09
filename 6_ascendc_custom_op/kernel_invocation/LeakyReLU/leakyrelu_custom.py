import os
import stat
import numpy as np
OPEN_FILE_MODES_640 = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP
WRITE_FILE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
np.random.seed(0)

def gen_golden_data_simple():
    total_length_imm = 8 * 200 * 1024
    tile_num_imm = 8

    total_length = np.array(total_length_imm, dtype=np.uint32)
    tile_num = np.array(tile_num_imm, dtype=np.uint32)
    scalar = np.array(0.1,dtype=np.float32)
    tiling = (total_length,tile_num,scalar)
    tiling_data = b''.join(x.tobytes() for x in tiling)

    with os.fdopen(os.open('./input/tiling.bin',WRITE_FILE_FLAGS, OPEN_FILE_MODES_640),"wb") as f:
        f.write(tiling_data)
    input_x = np.random.uniform(-100, 100, [8, 200, 1024]).astype(np.float16)
    golden = np.where(input_x > 0, input_x, input_x * scalar).astype(np.float16)
    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")


if __name__ == '__main__':
    gen_golden_data_simple()