import numpy as np

arr = np.array([0, 1, 2, 3, -4, 5, 6, 7, -8])
print(arr)

# 绝对值之和
print("1-范数：", np.linalg.norm(arr, ord=1))

# 平方和开根号
print("2-范数：", np.linalg.norm(arr, ord=2))

# 三次方和开三次根
print("3-范数：", np.linalg.norm(arr, ord=3))

# 绝对值的最大值
print("∞-范数：", np.linalg.norm(arr, ord=np.inf))
