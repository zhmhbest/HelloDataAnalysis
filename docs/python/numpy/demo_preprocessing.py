import numpy as np


if __name__ == '__main__':
    # 固定测试数据
    nd_arr = np.array([
        [-2, 2, 1, np.inf, np.nan, -3, np.inf, 1, -3, np.inf],
        [4, -4, 3, 0, 4, np.inf, -3, -4, -3, np.nan],
        [-2, 3, 1, 1, 3, 0, 4, 0, -3, -3],
        [-2, -3, 3, -4, np.nan, -1, 4, -2, np.nan, 1],
        [np.inf, 2, np.inf, np.nan, -2, 0, -3, np.nan, 2, -4]
    ])
    print(nd_arr)
    '''
    [[-2.  2.  1. inf nan -3. inf  1. -3. inf]
     [ 4. -4.  3.  0.  4. inf -3. -4. -3. nan]
     [-2.  3.  1.  1.  3.  0.  4.  0. -3. -3.]
     [-2. -3.  3. -4. nan -1.  4. -2. nan  1.]
     [inf  2. inf nan -2.  0. -3. nan  2. -4.]]
    '''

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
