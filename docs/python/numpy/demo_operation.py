import numpy as np


if __name__ == '__main__':
    """
        ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
        类型转换
    """
    nd_arr_i32 = np.array(1, dtype=np.int32)
    print(nd_arr_i32.dtype)
    # int32
    nd_arr_i16 = nd_arr_i32.astype('int16')
    print(nd_arr_i16.dtype)
    # int16
    nd_arr_i8 = nd_arr_i32.astype(np.int8)
    print(nd_arr_i8.dtype)
    # int8

    """
        ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
        Shape调整
    """
    shape_arr = np.arange(6)
    print(shape_arr)
    '''
    [0 1 2 3 4 5]
    '''

    print(shape_arr.reshape((1, 6)))
    print(shape_arr.reshape((1, -1)))
    print(shape_arr.reshape((-1, 6)))
    print(shape_arr[None, :])
    print(shape_arr[np.newaxis, :])
    '''
    [[0 1 2 3 4 5]]
    '''

    print(shape_arr.reshape((2, 3)))
    print(shape_arr.reshape((-1, 3)))
    print(shape_arr.reshape((2, -1)))
    '''
    [[0 1 2]
     [3 4 5]]
    '''

    print(shape_arr.reshape((6, 1)))
    print(shape_arr.reshape((6, -1)))
    print(shape_arr.reshape((-1, 1)))
    print(shape_arr[:, np.newaxis])
    print(shape_arr[:, None])
    '''
    [[0]
     [1]
     [2]
     [3]
     [4]
     [5]]
    '''

    shape_arr = np.arange(6).reshape(2, 3)
    print(shape_arr)
    '''
    [[0 1 2]
     [3 4 5]]
    '''
    print(shape_arr.reshape(6))
    print(shape_arr.reshape(-1))
    print(shape_arr.reshape([6]))
    print(shape_arr.reshape((6,)))
    '''
    [0 1 2 3 4 5]
    '''

    """
        ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
        Squeeze: 从ndarray中删除一维项
    """
    squeeze_arr1 = np.arange(6).reshape((2, 1, 3))
    print(squeeze_arr1)
    '''
    [[[0 1 2]]
     [[3 4 5]]]
    '''
    print(np.squeeze(squeeze_arr1))
    '''
    [[0 1 2]
     [3 4 5]]
    '''

    squeeze_arr2 = np.arange(6).reshape((1, 2, 3))
    print(squeeze_arr2)
    '''
    [[[0 1 2]
      [3 4 5]]]
    '''
    print(np.squeeze(squeeze_arr2))
    '''
    [[0 1 2]
     [3 4 5]]
    '''

    """
        ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
        Unsqueeze / expand_dims: 在ndarray中增加一维
    """
    expand_arr = np.arange(3)
    print(expand_arr)
    '''
    [0 1 2]
    '''

    print(np.expand_dims(expand_arr, 0))
    '''
    [[0 1 2]]
    '''

    print(np.expand_dims(expand_arr, 1))
    '''
    [[0]
     [1]
     [2]]
    '''

    """
        ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
        Stack拼接
    """
    stack_arr1 = np.arange(0, 6).reshape(2, -1)
    stack_arr2 = np.arange(6, 12).reshape(2, -1)
    print(stack_arr1)
    print(stack_arr2)
    '''
    [[0 1 2]
     [3 4 5]]
    [[ 6  7  8]
     [ 9 10 11]]
    '''

    # 垂直拼接
    print(np.vstack((stack_arr1, stack_arr2)))
    '''
    [[ 0  1  2]
     [ 3  4  5]
     [ 6  7  8]
     [ 9 10 11]]
    '''

    # 水平拼接
    print(np.hstack((stack_arr1, stack_arr2)))
    '''
    [[ 0  1  2  6  7  8]
     [ 3  4  5  9 10 11]]
    '''

    """
        ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
        转置
    """
    transpose_arr1 = np.array([
        [1, 2],
        [3, 4]
    ])
    print(transpose_arr1.T)
    print(np.transpose(transpose_arr1))
    print(np.swapaxes(transpose_arr1, 1, 0))
    print(np.swapaxes(transpose_arr1, 0, 1))
    '''
    [[1 3]
     [2 4]]
    '''

    transpose_arr2 = np.array([
        [[1, 2],
         [3, 4]],
        [[5, 6],
         [7, 8]],
    ])

    print(np.swapaxes(transpose_arr2, -1, -2))
    print(np.swapaxes(transpose_arr2, -2, -1))
    '''
    [[[1 3]
      [2 4]]
     [[5 7]
      [6 8]]]
    '''
