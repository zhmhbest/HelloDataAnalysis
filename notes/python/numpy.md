<link rel="stylesheet" href="https://zhmhbest.gitee.io/hellomathematics/style/index.css">
<script src="https://zhmhbest.gitee.io/hellomathematics/style/index.js"></script>

# [Numpy](../index.html)

[TOC]

```py
import numpy as np
```

## 数据类型

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

## 生成数据

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
        num     : 要生成的样本数
        endpoint: 是否包含stop
        retstep : 是否返回间距值
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

### 数据填充

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
    生成 [low, high) 内的随机整数。
    np.randint(low, high=None, size=None, dtype='l')
"""
nd_random1 = np.random.randint(10, size=(2, 3))
print(nd_random1)
'''
[[3 2 2]
 [9 4 6]]
'''
# 👆 实际运行结果不一定与此处一致


nd_random2 = np.random.randint(0, 10, size=(2, 3))
print(nd_random2)
'''
[[6 0 4]
 [4 8 3]]

'''
# 👆 实际运行结果不一定与此处一致
```

```py
"""
    0~1 内给定形状的随机值。
    rand(d0, d1, ..., dn)
"""
nd_random_rand1 = np.random.rand(2, 3)
print(nd_random_rand1)
'''
[[0.75515944 0.34902188 0.39546153]
 [0.60638465 0.67093481 0.38721671]]
'''
# 👆 实际运行结果不一定与此处一致

```

```py
"""
    从标准正态分布返回一个随机样本
    randn(d0, d1, ..., dn)
"""
nd_random_randn1 = np.random.randn(2, 3)
print(nd_random_randn1)
'''
[[ 0.71655872  0.86853771 -0.66122039]
 [-2.13445788 -0.85997662  0.09589859]]
'''
# 👆 实际运行结果不一定与此处一致
```

```py
"""
    从指定正态分布返回一个随机样本
    normal(loc=0.0, scale=1.0, size=None)
        loc  : 分布中心
        scale: 标准差
"""
nd_random_normal1 = np.random.normal(20, 0.6, (2, 3))
print(nd_random_normal1)
'''
[[19.71652233 20.25153326 20.89270548]
 [20.57732553 20.08643319 20.14295213]]
'''
# 👆 实际运行结果不一定与此处一致
```

```py
"""
    从均匀分布 [low, high) 中抽取样本
    uniform(low=0.0, high=1.0, size=None)
"""
nd_random_uniform1 = np.random.uniform(0, 10, size=(2, 3))
print(nd_random_uniform1)
'''
[[6.49230307 0.50673769 9.7944549 ]
 [8.74150152 4.47099043 4.15513917]]
'''
# 👆 实际运行结果不一定与此处一致
```

## 调整维度

### 序列升维

```py
# 测试数据
nd_arr = np.arange(24)
print(nd_arr, end='\n\n')
'''
    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
'''

print(nd_arr.reshape((1, 24)))
print(nd_arr.reshape((1, -1)))
print(nd_arr.reshape((-1, 24)))
print(nd_arr[None, :])
print(nd_arr[np.newaxis, :])
print()
'''
    [[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]]
'''

print(nd_arr.reshape((2, 12)))
print(nd_arr.reshape((-1, 12)))
print(nd_arr.reshape((2, -1)))
print()
'''
[[ 0  1  2  3  4  5  6  7  8  9 10 11]
 [12 13 14 15 16 17 18 19 20 21 22 23]]
'''

print(nd_arr.reshape((24, 1)))
print(nd_arr.reshape((24, -1)))
print(nd_arr.reshape((-1, 1)))
print(nd_arr[:, np.newaxis])
print(nd_arr[:, None])
'''
[[ 0]
 [ 1]
 ...
 [22]
 [23]]
'''
```

### 序列降维

```py
# 测试数据
nd_arr = np.arange(24).reshape((3, -1))
print(nd_arr, end='\n\n')
'''
[[ 0  1  2  3  4  5  6  7]
 [ 8  9 10 11 12 13 14 15]
 [16 17 18 19 20 21 22 23]]
'''

print(nd_arr.reshape(24))
print(nd_arr.reshape(-1))
print(nd_arr.reshape([24]))
print(nd_arr.reshape((24,)))
'''
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
'''
```

## 类型转换

```py
nd_arr = np.array(1, dtype=np.int32)
print(nd_arr.dtype)
# int32

nd_arr_i16 = nd_arr.astype('int16')
print(nd_arr_i16.dtype)
# int16

nd_arr_i8 = nd_arr.astype(np.int8)
print(nd_arr_i8.dtype)
# int8
```

## 数据拼接

```py
arr1 = np.arange(0, 10).reshape(2, -1)
print(arr1)
'''
[[0 1 2 3 4]
 [5 6 7 8 9]]
'''
arr2 = np.arange(10, 20).reshape(2, -1)
print(arr2)
'''
[[10 11 12 13 14]
 [15 16 17 18 19]]
'''

# 垂直拼接
print(np.vstack((arr1, arr2)))
'''
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]]
'''

# 水平拼接
print(np.hstack((arr1, arr2)))
'''
[[ 0  1  2  3  4 10 11 12 13 14]
 [ 5  6  7  8  9 15 16 17 18 19]]
'''
```

## 切片

**数据准备**

```py
nd_arr = np.arange(0, 60, dtype=np.int32).reshape(-1, 10)
print(nd_arr, nd_arr.shape)
'''
[[ 0  1  2  3  4  5  6  7  8  9]
 [10 11 12 13 14 15 16 17 18 19]
 [20 21 22 23 24 25 26 27 28 29]
 [30 31 32 33 34 35 36 37 38 39]
 [40 41 42 43 44 45 46 47 48 49]
 [50 51 52 53 54 55 56 57 58 59]] (6, 10)
'''
```

**行切片**

```py
# 任取一行
print(nd_arr[3])        # row = 3
print(nd_arr[3, :])     # row = 3
'''
[30 31 32 33 34 35 36 37 38 39]
'''

# 某行开始到结尾
print(nd_arr[3:])       # row = [3, *]
print(nd_arr[3:, :])    # row = [3, *]
'''
[[30 31 32 33 34 35 36 37 38 39]
 [40 41 42 43 44 45 46 47 48 49]
 [50 51 52 53 54 55 56 57 58 59]]
'''

# 某行开始到某行结束
print(nd_arr[1:4])      # row = [1, 4)
print(nd_arr[1:4, :])   # row = [1, 4)
'''
[[10 11 12 13 14 15 16 17 18 19]
 [20 21 22 23 24 25 26 27 28 29]
 [30 31 32 33 34 35 36 37 38 39]]
'''

# 某行开始到某行结束，有跨度
print(nd_arr[1:6:2])         # row = [1, 6) step 2 = 1, 3, 5
print(nd_arr[1:6:2, :])      # row = [1, 6) step 2 = 1, 3, 5
'''
[[10 11 12 13 14 15 16 17 18 19]
 [30 31 32 33 34 35 36 37 38 39]
 [50 51 52 53 54 55 56 57 58 59]]
'''

# 选取多行
print(nd_arr[[0, 2, 4]])        # row = 0, 2, 4
print(nd_arr[[0, 2, 4], :])     # row = 0, 2, 4
'''
[[ 0  1  2  3  4  5  6  7  8  9]
 [20 21 22 23 24 25 26 27 28 29]
 [40 41 42 43 44 45 46 47 48 49]]
'''
```

**列切片**

```py
# 取一列
print(nd_arr[:, 3])             # col = 3
'''
[ 3 13 23 33 43 53]
'''

# 某列开始到结尾
print(nd_arr[:, 7:])            # col = [7, *]
'''
[[ 7  8  9]
 ...
 [57 58 59]]
'''

# 某列开始到某列结束
print(nd_arr[:, 1:4])           # col = [1, 4)
'''
[[ 1  2  3]
 ...
 [51 52 53]]
'''

# 某列开始到某列结束，有跨度
print(nd_arr[:, 1:6:2])         # col = [1, 6) step 2 = 1, 3, 5
'''
[[ 1  3  5]
 ...
 [51 53 55]]
'''

# 选取多列
print(nd_arr[:, [0, 2, 4]])     # col = 0, 2, 4
'''
[[ 0  2  4]
 ...
 [50 52 54]]
'''
```

## 读取数据

```py
def sequence_generator(row_num, delimiter=','):
    for _i in range(row_num):
        if 0 == _i % 3:
            yield '#' + delimiter.join([str(it) for it in range(_i*10, (_i+1)*10)])
        else:
            yield delimiter.join([str(it) for it in range(_i*10, (_i+1)*10)])
# end def

for it in sequence_generator(5):
    print(it)
'''
#0,1,2,3,4,5,6,7,8,9
10,11,12,13,14,15,16,17,18,19
20,21,22,23,24,25,26,27,28,29
#30,31,32,33,34,35,36,37,38,39
40,41,42,43,44,45,46,47,48,49
'''
```

```py
"""
    np.loadtxt(
        fname,              # {str} 文件名，文件类型可以是text, gz, bz2
        dtype=float,
        comments='#',       # {str} 注释符
        delimiter=None,     # {str} 数据分割符，默认为 space
        converters=None,    # {dict} 用于调整列数据
        skiprows=0,         # {int} 跳过开头几行，默认为 0
        usecols=None,       # {sequence} 指定读取哪几行，默认读取全部
        unpack=False,       # {bool} 是否对数据进行转置
        ndmin=0,            # {int} 
        encoding='bytes',   # {str} 文件编码
        max_rows=None       # {int} 最多读取行数（不算跳过的部分），默认无限制
)
"""

print(np.loadtxt(sequence_generator(5), dtype=np.int, delimiter=','))
'''
[[10 11 12 13 14 15 16 17 18 19]
 [20 21 22 23 24 25 26 27 28 29]
 [40 41 42 43 44 45 46 47 48 49]]
'''

print(np.loadtxt(sequence_generator(5), skiprows=2, dtype=np.int, delimiter=','))
'''
[[20 21 22 23 24 25 26 27 28 29]
 [40 41 42 43 44 45 46 47 48 49]]
'''

print(np.loadtxt(sequence_generator(5), usecols=(2, 3), dtype=np.int, delimiter=','))
'''
[[12 13]
 [22 23]
 [42 43]]
'''
```

## 数据分析

>- `axis=0`：按列统计，结果为行
>- `axis=1`：按行统计，结果为列

### 数据预处理

```py
def generate_test_data():
    nd_arr = np.random.randint(-6, 5, size=(5, 10))
    nd_arr = np.where(nd_arr == -6, np.nan, nd_arr)
    nd_arr = np.where(nd_arr == -5, np.inf, nd_arr)
    print(nd_arr, end='\n\n')
    return nd_arr
    # 取值范围：nan inf -4 -3 -2 -1 0 1 2 3 4


# 固定测试数据
# nd_arr = generate_test_data()
nd_arr = np.array([
    [-2, 2, 1, np.inf, np.nan, -3, np.inf, 1, -3, np.inf],
    [4, -4, 3, 0, 4, np.inf, -3, -4, -3, np.nan],
    [-2, 3, 1, 1, 3, 0, 4, 0, -3, -3],
    [-2, -3, 3, -4, np.nan, -1, 4, -2, np.nan, 1],
    [np.inf, 2, np.inf, np.nan, -2, 0, -3, np.nan, 2, -4]
])

# 统计np.nan数量
print('nan in col', np.count_nonzero(np.isnan(nd_arr), axis=0))
print('nan in row', np.count_nonzero(np.isnan(nd_arr), axis=1))
'''
nan in col [0 0 0 1 2 0 0 1 1 1]
nan in row [1 1 0 2 2]
'''

# 统计np.inf数量
print('inf in col', np.count_nonzero(np.isinf(nd_arr), axis=0))
print('inf in row', np.count_nonzero(np.isinf(nd_arr), axis=1))
'''
inf in col [1 0 1 1 0 1 1 0 0 1]
inf in row [3 1 0 0 2]
'''

# 以0填充nan，以99填充inf
# nd_arr = np.where(np.isnan(nd_arr), 0, nd_arr)
nd_arr[np.isnan(nd_arr)] = 0
# nd_arr = np.where(np.isinf(nd_arr), 99, nd_arr)
nd_arr[np.isinf(nd_arr)] = 99
print(nd_arr)
'''
[[-2.  2.  1. 99.  0. -3. 99.  1. -3. 99.]
 [ 4. -4.  3.  0.  4. 99. -3. -4. -3.  0.]
 [-2.  3.  1.  1.  3.  0.  4.  0. -3. -3.]
 [-2. -3.  3. -4.  0. -1.  4. -2.  0.  1.]
 [99.  2. 99.  0. -2.  0. -3.  0.  2. -4.]]
'''

# 数据裁剪.超出限制的数据会被替换为边界值
nd_arr = np.clip(nd_arr, -2, 2)
print(nd_arr)
'''
[[-2.  2.  1.  2.  0. -2.  2.  1. -2.  2.]
 [ 2. -2.  2.  0.  2.  2. -2. -2. -2.  0.]
 [-2.  2.  1.  1.  2.  0.  2.  0. -2. -2.]
 [-2. -2.  2. -2.  0. -1.  2. -2.  0.  1.]
 [ 2.  2.  2.  0. -2.  0. -2.  0.  2. -2.]]
'''

# 统计大于0的数量
print(np.count_nonzero(nd_arr > 0, axis=0))
print(np.count_nonzero(nd_arr > 0, axis=1))
'''
[2 3 5 2 2 1 3 1 1 2]
[6 4 5 3 4]
'''
```

### 转置

```py
nd_arr = np.array([
    [1, 2],
    [3, 4]
])

print(nd_arr.T)
print(np.transpose(nd_arr))
print(np.swapaxes(nd_arr, 1, 0))
print(np.swapaxes(nd_arr, 0, 1))
'''
[[1 3]
 [2 4]]
'''
```

### 数据统计

```py
nd_arr = np.array([
    [70, 64, 37, 24, 82, 67, 56, 56, 22, 79],
    [12, 22, 10, 37, 83, 42, 95, 36, 16, 62],
    [93, 84, 62, 51, 82, 99, 36, 46, 60, 84]
])
print(nd_arr, end='\n\n')

# 最大值
print(np.max(nd_arr, axis=0))
print(np.max(nd_arr, axis=1))
print(np.max(nd_arr))
'''
[93 84 62 51 83 99 95 56 60 84]
[82 95 99]
99
'''

# 最小值
print(np.min(nd_arr, axis=0))
print(np.min(nd_arr, axis=1))
print(np.min(nd_arr))
'''
[12 22 10 24 82 42 36 36 16 62]
[22 10 36]
10
'''

# 求和
print(np.sum(nd_arr, axis=0))
print(np.sum(nd_arr, axis=1))
print(np.sum(nd_arr))
'''
[175 170 109 112 247 208 187 138  98 225]
[557 415 697]
1669
'''

# 中位数
print(np.median(nd_arr, axis=0))
print(np.median(nd_arr, axis=1))
print(np.median(nd_arr))
'''
[70. 64. 37. 37. 82. 67. 56. 46. 22. 79.]
[60.  36.5 72. ]
58.0
'''

# 平均数
print(np.mean(nd_arr, axis=0, dtype=np.int32))
print(np.mean(nd_arr, axis=1, dtype=np.int32))
print(np.mean(nd_arr, dtype=np.int32))
'''
[58 56 36 37 82 69 62 46 32 75]
[55 41 69]
55
'''

# 加权平均数
w = np.array([
    [7, 1, 8, 1, 4, 2, 5, 7, 5, 9],
    [5, 8, 4, 7, 2, 3, 6, 3, 2, 2],
    [7, 3, 6, 6, 6, 6, 2, 1, 8, 9]
])  # 👈每个数据对应的权值
print(np.average(nd_arr, axis=0, weights=w))
print(np.average(nd_arr, axis=1, weights=w))
print(np.average(nd_arr, weights=w))
'''
[63.21052632 41.         39.33333333 42.07142857 82.16666667 77.63636364
 70.92307692 49.63636364 41.46666667 79.55      ]
[57.73469388 39.54761905 74.46296296]
58.69655172413793
'''

# 极差（最大-最小）
print(np.ptp(nd_arr, axis=0))
print(np.ptp(nd_arr, axis=1))
print(np.ptp(nd_arr))
'''
[81 62 52 27  1 57 59 20 44 22]
[60 85 63]
89
'''

# 方差
print(np.var(nd_arr, axis=0))
print(np.var(nd_arr, axis=1))
print(np.var(nd_arr))
'''
[1.16155556e+03 6.67555556e+02 4.50888889e+02 1.21555556e+02
 2.22222222e-01 5.44222222e+02 6.00222222e+02 6.66666667e+01
 3.79555556e+02 8.86666667e+01]
[412.61 794.85 416.21]
673.7655555555555
'''

# 标准差
print(np.std(nd_arr, axis=0))
print(np.std(nd_arr, axis=1))
print(np.std(nd_arr))
'''
[34.08160142 25.8370965  21.23414441 11.02522361  0.47140452 23.32857094
 24.4994331   8.16496581 19.48218559  9.41629793]
[20.31280384 28.19308426 20.40122545]
25.956994347488607
'''
```
