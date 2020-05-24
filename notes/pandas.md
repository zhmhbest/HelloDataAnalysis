<link rel="stylesheet" href="https://zhmhbest.gitee.io/hellomathematics/style/index.css">
<script src="https://zhmhbest.gitee.io/hellomathematics/style/index.js"></script>

# [Pandas](./index.html)

>[Pandas API](https://pandas.pydata.org/pandas-docs/stable/reference/index.html)

[TOC]

```py
import pandas as pd
```

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

### 创建

```py
"""
    带有轴标签的一维ndarray
    Series(data=None, index=None, dtype=None, name=None, copy=False)
        data: {list|iter|dict|scalar} 数据
        index: {list} 索引
"""

print(pd.Series(['a', 'b', 'c'], range(3)))
'''
0    a
1    b
2    c
dtype: object
'''

print(pd.Series({'a': 1, 'b': 2, 'c': 3}, dtype='int64'))
'''
a    1
b    2
c    3
dtype: int64
'''

print(pd.Series(range(3)))
'''
0    0
1    1
2    2
dtype: int64
'''
```

### 使用

```py
ser1 = pd.Series({'a': 10, 'b': 11, 'c': 12})
ser2 = pd.Series({'b': 21, 'c': 22, 'D': 23})

# 按索引名称取值
print(ser1['b'])
# 11

# 按索引顺序取值
print(ser1.values[1])
# 11

# 取索引
print(ser1.index)
# Index(['a', 'b', 'c'], dtype='object')

# 修改索引
ser1.index = ['A', 'b', 'c']
print(ser1)
'''
A    10
b    11
c    12
dtype: int64
'''

# 自动对齐
print(ser1+ser2)
'''
A     NaN
D     NaN
b    32.0
c    34.0
dtype: float64
'''
```

## DataFrame

>[API DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/frame.html)

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