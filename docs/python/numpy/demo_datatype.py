import numpy as np


if __name__ == '__main__':
    numpy_types = [
        # np.bool,
        np.bool_,
        np.int_,        # auto long  ; 64 = int32
        np.intc,        # auto int   ; 64 = int32
        np.intp,        # auto size_t; 64 = int64

        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,

        np.float_,      # auto float; 64 = float64
        np.float16,
        np.float32,
        np.float64,

        np.complex_,    # auto complex; 64 = complex128
        np.complex64,   # 双 32 位浮点数
        np.complex128,  # 双 64 位浮点数

        np.nan,
        np.inf          # 无穷
    ]
    for i in numpy_types:
        print(i)
    # end for

    # nan之间互不相等
    print(np.nan == np.nan)
    print(np.isnan(np.nan))
