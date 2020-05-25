<link rel="stylesheet" href="https://zhmhbest.gitee.io/hellomathematics/style/index.css">
<script src="https://zhmhbest.gitee.io/hellomathematics/style/index.js"></script>

# [Pandas](./index.html)

[TOC]

```py
import numpy as np
import pandas as pd
```

[`Pandas API`](https://pandas.pydata.org/pandas-docs/stable/reference/index.html)

## Types

| Python | Pandas | Numpy |
| - | - | - |
| `str`      | `'object'`     | `np.string_`、`np.unicode_` |
| `int`      | `'int64'`      | `np.int64` |
| `float`    | `'float64'`    | `np.float64` |
| `bool`     | `'bool'`       | `np.bool_` |
| `datetime` | `'datetime64'` | N/A |

## Series

>[API Series](https://pandas.pydata.org/pandas-docs/stable/reference/series.html)

**测试数据**

```py
ser1 = pd.Series({'a': 10, 'b': 11, 'c': 12}, dtype='int32')
```

### \_\_init\_\_

```py
"""
    带有轴标签的一维ndarray
    Series(data=None, index=None, dtype=None, name=None, copy=False)
        data : {list|Iterable|dict|scalar} 数据
        index: {list} 索引
        name : {str} Series的名称
"""

# list
print(pd.Series(data=['a', 'b', 'c'], index=range(3), name='ser_list'))
'''
0    a
1    b
2    c
Name: ser_list, dtype: object
'''

# dict
print(pd.Series(data={'a': 1, 'b': 2, 'c': 3}, dtype='int64', name='ser_dict'))
'''
a    1
b    2
c    3
Name: ser_dict, dtype: int64
'''

# Iterable
print(pd.Series(data=range(3)))
'''
0    0
1    1
2    2
dtype: int64
'''
```

### Attributes

```py
# 索引
print(ser1.index)
# Index(['a', 'b', 'c'], dtype='object')

# 类型
print(ser1.dtype)
# int32

# 元素个数
print(ser1.size)
# 3
```

### Select

```py
# 开头
print(ser1.head(2))
'''
a    10
b    11
dtype: int32
'''

# 结尾
print(ser1.tail(2))
'''
b    11
c    12
dtype: int32
'''
```

```py
# 按索引名称取值
print(ser1['b'])
print(ser1.get('b'))
print(ser1.loc['b'])
print(ser1.at['b'])
# 11

# 按索引顺序取值
print(ser1.iloc[1])
print(ser1.iat[1])
# 11
```

## DataFrame

>[API DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/frame.html)

**测试数据**

```py
def generate_test_df(row_num, col_num):
    return pd.DataFrame(
        data=np.arange(row_num*col_num).reshape(row_num, col_num),
        index=[chr(97+i) for i in range(row_num)],
        columns=[chr(65+i) for i in range(col_num)],
        dtype='int32'
    )


df_test = generate_test_df(5, 10)
print(df_test)
'''
    A   B   C   D   E   F   G   H   I   J
a   0   1   2   3   4   5   6   7   8   9
b  10  11  12  13  14  15  16  17  18  19
c  20  21  22  23  24  25  26  27  28  29
d  30  31  32  33  34  35  36  37  38  39
e  40  41  42  43  44  45  46  47  48  49
'''
```

### \_\_init\_\_

```py
"""
    二维、大小可变、潜在异构的表格数据
    pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)
        data    : {dict|list of list|list of Series|DataFrame}
        index   : {list} 指定索引
        columns : {list} 指定列名
        dtype   : {None|str|np.dtypes} 指定数据类型，None表示自动推断
"""

# list of list: 需要指定索引和列名
df_lol = pd.DataFrame([
    [11, 21],
    [12, 22]
], index=['row_1', 'row_2'], columns=['col_1', 'col_2'])
print(df_lol)
'''
       col_1  col_2
row_1     11     21
row_2     12     22
'''

# dict: 自带列名，需要指定索引
df_dict = pd.DataFrame({
    'col_1': [11, 12],
    'col_2': [21, 22]
}, index=['row_1', 'row_2'])
print(df_dict)
'''
       col_1  col_2
row_1     11     21
row_2     12     22
'''

# Series: Series.name=索引，Series.index=列名
df_los = pd.DataFrame([
    pd.Series([11, 12], name='row_1', index=['col_1', 'col_2']),
    pd.Series([21, 22], name='row_2', index=['col_1', 'col_2'])
])
print(df_los)
'''
       col_1  col_2
row_1     11     21
row_2     12     22
'''
```

### Attributes

```py
print("元素总数：", df_test.size)
print("维度形状：", df_test.shape)
print("维度深度：", df_test.ndim)

print("行列索引：", df_test.axes)
print("行索引：", df_test.index)
print("列索引：", df_test.columns)

print("每列元素类型：", ','.join([str(i) for i in df_test.dtypes]))
print("每列占用内存：", ','.join([str(i) for i in df_test.memory_usage(index=False)]))
print("index占用内存：",  df_test.memory_usage()[0])
print("DF是否为空：",  df_test.empty)

'''
元素总数： 50
维度形状： (5, 10)
维度深度： 2
行列索引： [Index(['a', 'b', 'c', 'd', 'e'], dtype='object'), Index(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], dtype='object')]
行索引： Index(['a', 'b', 'c', 'd', 'e'], dtype='object')
列索引： Index(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], dtype='object')
每列元素类型： int32,int32,int32,int32,int32,int32,int32,int32,int32,int32
每列占用内存： 20,20,20,20,20,20,20,20,20,20
index占用内存： 40
DF是否为空： False
'''
```

### Select

**开头、结尾**

```py
# 开头n行
print(df_test.head(1))
'''
   A  B  C  D  E  F  G  H  I  J
a  0  1  2  3  4  5  6  7  8  9
'''

# 结尾n行
print(df_test.tail(1))
'''
    A   B   C   D   E   F   G   H   I   J
e  40  41  42  43  44  45  46  47  48  49
'''
```

**一列、一行**

```py
# 按col name，选择一列Series
print(df_test['A'])
print(df_test.get('A'))
'''
a     0
...
e    40
Name: A, dtype: int32
'''

# 按row name，选择一行Series
print(df_test.loc['a'])
'''
A    0
...
J    9
Name: a, dtype: int32
'''

# 按row index，选择一行Series
print(df_test.iloc[0])
'''
A    0
...
J    9
Name: a, dtype: int32
'''
```

**单元格**

```py
# 按(row name, col name)，选择一个单元格numpy.type
print(df_test.at['a', 'A'])
'''
0
'''

# 按(row index, col index)，选择一个单元格numpy.type
print(df_test.iat[0, 0])
'''
0
'''
```

### Update

```py

```

## IO

>[API IO](https://pandas.pydata.org/pandas-docs/stable/reference/io.html)

### CSV

>[API read_csv](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

```py
"""
    pd.read_csv(**)
        filepath_or_bufferstr: {str} 文件名
        sep                  : {str} 数据分割符，默认为','
        delimiter            : {str} sep的别名
        header               : {int|None} 记录列名的数据行，0为第1行，None为无记录
        names                : {list} 重新定义列名
        index_col            : {int|str|False} 索引列或无索引
        usecols              : {list} 使用的列
        squeeze              : {bool} 如果数据最终只有一列，则返回Series
        dtype                : {dict: col->type}
        skipinitialspace     : {bool} 在数据分割符后跳过空格
        skiprows             : {int|lambda x:} 跳过开头几行，或是否跳过某行
        skip_blank_lines     : {bool} 跳过空白行
        parse_dates          : {list: col_names} 指定日期列
        encoding             : {str} 建议'utf-8'
"""
```

>[API to_csv](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html)

```py
"""
    pd.DataFrame().to_csv(**)
        path_or_bufstr  : {str} 文件名
        sep             : {str} 分隔符
        columns         : {list} 写入的列
        header          : {list} 重新指定列名
        encoding        : {str} 建议'utf-8-sig'
"""
```

## DatetimeIndex

>[API DatetimeIndex](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.html)

<!-- ```py
    # 生成时间序列
    # start end : 时间起止
    # periods : 生成个数
    # freq : 生成频率
    """
    D : Day
    B : BusinessDay
    H : Hour
    T/min : Minutes
    S : Second
    L/ms : Milli
    U : Micro
    M : MonthEnd
    BM : BusinessMonthEnd
    MS : MonthBegin
    BMS : BusinessMonthBegin
    """
    print('时间序列')
    df = pd.date_range("1996-10-16", "2019-10-25", periods=None, freq='Y')
    print(len(df))
    print(df[0])
    print(df)
``` -->