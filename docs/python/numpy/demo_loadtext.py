import numpy as np


def sequence_generator(row_num, delimiter=','):
    for _i in range(row_num):
        if 0 == _i % 3:
            yield '#' + delimiter.join([str(it) for it in range(_i*10, (_i+1)*10)])
        else:
            yield delimiter.join([str(it) for it in range(_i*10, (_i+1)*10)])
# end def


if __name__ == '__main__':
    for it in sequence_generator(5):
        print(it)
    '''
    #0,1,2,3,4,5,6,7,8,9
    10,11,12,13,14,15,16,17,18,19
    20,21,22,23,24,25,26,27,28,29
    #30,31,32,33,34,35,36,37,38,39
    40,41,42,43,44,45,46,47,48,49
    '''

    """
        ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
        loadtxt(
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

    # 跳过注释
    print(np.loadtxt(sequence_generator(5), dtype=np.int, delimiter=','))
    '''
    [[10 11 12 13 14 15 16 17 18 19]
     [20 21 22 23 24 25 26 27 28 29]
     [40 41 42 43 44 45 46 47 48 49]]
    '''

    # 跳过开头2行（包含注释行）
    print(np.loadtxt(sequence_generator(5), skiprows=2, dtype=np.int, delimiter=','))
    '''
    [[20 21 22 23 24 25 26 27 28 29]
     [40 41 42 43 44 45 46 47 48 49]]
    '''

    # 限制返回的列
    print(np.loadtxt(sequence_generator(5), usecols=(2, 3), dtype=np.int, delimiter=','))
    '''
    [[12 13]
     [22 23]
     [42 43]]
    '''
