import numpy as np


if __name__ == '__main__':
    """
        ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
        array(object, dtype=None, copy=True)
    """
    nd_arr = np.array([1, 2, 3, 4])
    print(type(nd_arr))
    print(nd_arr, nd_arr.dtype)

    """
        ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
         arange([start,] stop[, step,], dtype=None)
    """
    range_arr1 = np.arange(5)
    print(range_arr1, range_arr1.dtype)
    # [0 1 2 3 4] int32

    range_arr2 = np.arange(0, 5)
    print(range_arr2, range_arr2.dtype)
    # [0 1 2 3 4] int32

    range_arr3 = np.arange(0, 5, 1, dtype=np.int8)
    print(range_arr3, range_arr3.dtype)
    # [0 1 2 3 4] int8

    """
        ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
        返回指定间隔上的等距数字
        linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None):
            num     : 要生成的样本数
            endpoint: 是否包含stop
            retstep : 是否返回间距值
    """
    line_arr1 = np.linspace(0, 100, num=10, endpoint=False)
    print(line_arr1)
    # [ 0. 10. 20. 30. 40. 50. 60. 70. 80. 90.]

    line_arr2 = np.linspace(0, 100, num=11, endpoint=True)
    print(line_arr2)
    # [  0.  10.  20.  30.  40.  50.  60.  70.  80.  90. 100.]

    """
        ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
        返回在对数刻度上均匀分布的数字
        logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)
            num     : 要生成的样本数
            endpoint: 是否包含stop
            base    : 对数空间的基
    """
    line_arr1 = np.logspace(0, 9, num=10, base=2)
    print(line_arr1)
    # [  1.   2.   4.   8.  16.  32.  64. 128. 256. 512.]

    """
        ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
        eye(N, M=None, k=0, dtype=float, order='C')
            N     : 行
            M     : 列
            k     : {0 主对角线 | + 上对角线 | - 下对角线}
            order : {'C' 按行存储 | 'F' 按列存储}
    """
    print(np.eye(3, 4, k=0))
    print(np.eye(3, 4, k=1))
    print(np.eye(3, 4, k=-1))
    '''
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]]
    [[0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]
    [[0. 0. 0. 0.]
     [1. 0. 0. 0.]
     [0. 1. 0. 0.]]
    '''

    """
        ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
        返回主对角线为1的方形矩阵
        identity(n, dtype=None)
    """
    print(np.identity(3))
    '''
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]
    '''

    """
        ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
        返回以1填充的ndarray
        ones(shape, dtype=None, order='C')
    """
    print(np.ones(3))
    print(np.ones((3, 4)))
    '''
    [1. 1. 1.]
    [[1. 1. 1. 1.]
     [1. 1. 1. 1.]
     [1. 1. 1. 1.]]
    '''

    """
        ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
        返回以0填充的ndarray
        zeros(shape, dtype=None, order='C')
    """
    print(np.zeros(3))
    print(np.zeros((3, 4)))
    '''
    [0. 0. 0.]
    [[0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]]
    '''

    """
        ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
        矩阵内的值未初始化，直接使用就是内存碎片的值
        empty(shape, dtype=None, order='C')
    """
    print(np.empty((3, 4)))
    '''
    [[0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]]
    '''
    # ↑↑↑ 实际运行结果不一定相同 ↑↑↑
