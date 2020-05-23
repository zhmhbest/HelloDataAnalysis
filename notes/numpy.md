<link rel="stylesheet" href="https://zhmhbest.gitee.io/hellomathematics/style/index.css">
<script src="https://zhmhbest.gitee.io/hellomathematics/style/index.js"></script>

# [Numpy](./index.html)

[TOC]

```py
import numpy as np
```

## types

```py
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
```

## ndarray

### 固定数组

```py
"""
    array(object, dtype=None, copy=True)
"""
nd_arr = np.array([1, 2, 3, 4])
print(type(nd_arr))
print(nd_arr, nd_arr.dtype)
```

### 序列生成器

```py
"""
    返回给定间隔内的等间距值
    np.arange(start, stop, step, dtype=None)
"""
arr1 = np.arange(5)
print(arr1, arr1.dtype)
# [0 1 2 3 4] int32

arr2 = np.arange(0, 5)
print(arr2, arr2.dtype)
# [0 1 2 3 4] int32

arr3 = np.arange(0, 5, 1, dtype=np.int8)
print(arr3, arr3.dtype)
# [0 1 2 3 4] int8
```

```py
"""
    返回指定间隔上的等距数字
    np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
        num: 要生成的样本数
        endpoint: 是否包含stop
        retstep: 是否返回间距值
"""
arr1 = np.linspace(0, 100, num=10, endpoint=False)
print(arr1)
# [ 0. 10. 20. 30. 40. 50. 60. 70. 80. 90.]

arr2 = np.linspace(0, 100, num=11, endpoint=True)
print(arr2)
# [  0.  10.  20.  30.  40.  50.  60.  70.  80.  90. 100.]
```

```py
"""
    返回在对数刻度上均匀分布的数字
    np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)
        base: 对数空间的基
"""
arr1 = np.logspace(0, 9, num=10, base=2)
print(arr1)
# [  1.   2.   4.   8.  16.  32.  64. 128. 256. 512.]


def log_space(start, stop, endpoint=True, base=10, dtype=np.float64):
    return np.array(
        [base**i for i in range(start, (1+stop if endpoint else stop))],
        dtype=dtype
    )


arr2 = log_space(0, 9, base=2)
print(arr2)
# [  1.   2.   4.   8.  16.  32.  64. 128. 256. 512.]
```

### 矩阵生成器

```py
mat_eye1 = np.eye(3, 4, k=0)
print(mat_eye1)
'''
[[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]]
'''

mat_eye2 = np.eye(3, 4, k=1)
print(mat_eye2)
'''
[[0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]]
'''
```

```py
mat_idt = np.identity(3)
print(mat_idt)
'''
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
'''
```

```py
mat_ones = np.ones((3, 4))
print(mat_ones)
'''
[[1. 1. 1. 1.]
 [1. 1. 1. 1.]
 [1. 1. 1. 1.]]
'''
```

```py
mat_zeros = np.zeros((3, 4))
print(mat_zeros)
'''
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]
'''
```

```py
# 矩阵内的值未初始化，直接使用就是内存碎片的值。
mat_empty = np.empty((3, 4))
print(mat_empty)
'''
[[6.23042070e-307 4.67296746e-307 1.42420209e-306 1.11260619e-306]
 [3.33776697e-307 1.69119330e-306 1.24610994e-306 7.56571288e-307]
 [1.33508845e-306 1.69122046e-306 1.69118108e-306 3.20558585e-317]]
'''
# 👆 实际运行结果不一定与此处一致
```

### 随机

```py
"""
    生成[low, high)内的随机整数。
    np.randint(low, high=None, size=None, dtype='l')
"""
nd_random1 = np.random.randint(10, size=(4, 5))
print(nd_random1)
'''
[[1 6 8 2 4]
 [8 0 1 9 1]
 [7 8 9 8 0]
 [6 3 0 0 7]]
'''
# 👆 实际运行结果不一定与此处一致


nd_random2 = np.random.randint(0, 10, size=(4, 5))
print(nd_random2)
'''
[[2 0 6 2 6]
 [5 0 3 9 9]
 [0 0 4 0 9]
 [3 3 5 1 9]]
'''
# 👆 实际运行结果不一定与此处一致
```

```py
"""
    给定形状中的随机值。
    rand(d0, d1, ..., dn)
"""
nd_random_rand1 = np.random.rand(3, 4)
print(nd_random_rand1)
'''
[[0.72896113 0.5653343  0.4512705  0.72704046]
 [0.74125539 0.70318518 0.73897613 0.59652132]
 [0.0965708  0.11740313 0.94253921 0.27898263]]
'''
# 👆 实际运行结果不一定与此处一致


"""
    从标准正态分布返回一个随机样本
    randn(d0, d1, ..., dn)
"""
nd_random_randn1 = np.random.randn(3, 4)
print(nd_random_randn1)
'''
[[-0.4542668  -1.97113359 -1.44459803  0.45791074]
 [-1.30713887 -0.28921407  0.59186834 -0.86553223]
 [ 0.28132592 -0.55796305  0.90973855  0.82795517]]
'''
# 👆 实际运行结果不一定与此处一致
```

## reshape

