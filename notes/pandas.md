<link rel="stylesheet" href="https://zhmhbest.gitee.io/hellomathematics/style/index.css">
<script src="https://zhmhbest.gitee.io/hellomathematics/style/index.js"></script>

# [Pandas](./index.html)

[TOC]

>[Pandas API](https://pandas.pydata.org/pandas-docs/stable/reference/index.html)

```py
import numpy as np
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

**筛选**

```py
# 筛选数据，未选到的就是NaN
print(df_test[0 == df_test % 2])
print(df_test.where(0 == df_test % 2, other=np.nan, inplace=False))
'''
    A   B   C   D   E   F   G   H   I   J
a   0 NaN   2 NaN   4 NaN   6 NaN   8 NaN
b  10 NaN  12 NaN  14 NaN  16 NaN  18 NaN
c  20 NaN  22 NaN  24 NaN  26 NaN  28 NaN
d  30 NaN  32 NaN  34 NaN  36 NaN  38 NaN
e  40 NaN  42 NaN  44 NaN  46 NaN  48 NaN
'''
```

### Update Shape

**删除**

```py
# 创建视图
df_test_row = df_test[:]

# 删除行
df_test_row = df_test_row.drop('a')
print(df_test_row)
'''
    A   B   C   D   E   F   G   H   I   J
b  10  11  12  13  14  15  16  17  18  19
c  20  21  22  23  24  25  26  27  28  29
d  30  31  32  33  34  35  36  37  38  39
e  40  41  42  43  44  45  46  47  48  49
'''

# 创建视图
df_test_col = df_test[:]

# 删除列
del df_test_col['A']

# 弹出列
col_B = df_test_col.pop('B')
print(col_B)
print(df_test_col)
'''
a     1
b    11
c    21
d    31
e    41
Name: B, dtype: int32

    C   D   E   F   G   H   I   J
a   2   3   4   5   6   7   8   9
b  12  13  14  15  16  17  18  19
c  22  23  24  25  26  27  28  29
d  32  33  34  35  36  37  38  39
e  42  43  44  45  46  47  48  49
'''
```

**追加行**

```py
df_test.loc['new_row1'] = range(90, 100)
df_test.loc['new_row2'] = pd.Series({
    'A': 0,
    'E': 0
})
df_test = df_test.append(pd.DataFrame([
    pd.Series([0, 1], name='new_row3', index=['G', 'D']),
    pd.Series([0, 1], name='new_row4', index=['F', 'C']),
]))
print(df_test)
'''
             A     B     C     D     E     F     G     H     I     J
a          0.0   1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0
b         10.0  11.0  12.0  13.0  14.0  15.0  16.0  17.0  18.0  19.0
c         20.0  21.0  22.0  23.0  24.0  25.0  26.0  27.0  28.0  29.0
d         30.0  31.0  32.0  33.0  34.0  35.0  36.0  37.0  38.0  39.0
e         40.0  41.0  42.0  43.0  44.0  45.0  46.0  47.0  48.0  49.0
new_row1  90.0  91.0  92.0  93.0  94.0  95.0  96.0  97.0  98.0  99.0
new_row2   0.0   NaN   NaN   NaN   0.0   NaN   NaN   NaN   NaN   NaN
new_row3   NaN   NaN   NaN   1.0   NaN   NaN   0.0   NaN   NaN   NaN
new_row4   NaN   NaN   1.0   NaN   NaN   0.0   NaN   NaN   NaN   NaN
'''
```

**追加列**

```py
df_test['new_col1'] = range(90, 95)
df_test['new_col2'] = pd.Series([0, 1], index=['a', 'b'])
df_test = df_test.assign(
    new_col3=pd.Series([0, 1], index=['b', 'c']),
    new_col4=pd.Series([0, 1], index=['c', 'd'])
)
print(df_test)
'''
    A   B   C   D   E   F  ...   I   J  new_col1  new_col2  new_col3  new_col4
a   0   1   2   3   4   5  ...   8   9        90       0.0       NaN       NaN
b  10  11  12  13  14  15  ...  18  19        91       1.0       0.0       NaN
c  20  21  22  23  24  25  ...  28  29        92       NaN       1.0       0.0
d  30  31  32  33  34  35  ...  38  39        93       NaN       NaN       1.0
e  40  41  42  43  44  45  ...  48  49        94       NaN       NaN       NaN
'''
```

### Calculate

```py
def calc_row(row):
    print(row.sum())


def calc_col(col):
    print(col.sum())


df_test.apply(calc_row, axis=1)
print(df_test.sum(axis=1))
'''
a     45
b    145
c    245
d    345
e    445
dtype: int64
'''

df_test.apply(calc_col, axis=0)
print(df_test.sum(axis=0))
'''
A    100
B    105
C    110
D    115
E    120
F    125
G    130
H    135
I    140
J    145
dtype: int64
'''
```

### Join



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