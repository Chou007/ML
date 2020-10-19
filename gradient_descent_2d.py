import math
import numpy as np

def func_2d(x):
    """
    x 是一个二维向量，
    构建目标函数
    """
    return -math.exp(-(x[0] ** 2 + x[1] ** 2))

def gradience_2d(x):
    """
    对目标函数梯度
    """
    grad1 = 2 * x[0] * math.exp(-(x[0] ** 2 + x[1] ** 2))
    grad2 = 2 * x[1] * math.exp(-(x[0] ** 2 + x[1] ** 2))
    return np.array([grad1, grad2])

def gradient_descent_2d(gradient, cur_x=np.array([0.1, 0.1]), learning_rate=0.01, precision=0.0001, max_iters=10000):
    """
    二维梯度下降：
    param gradient: 目标函数梯度
    param cur_x: 计算初始值
    param learning_rate:学习速率
    param precision:收敛精度
    param max_iter:最大迭代次数 
    """
    for i in range(max_iters):
        grad_cur = gradient(cur_x)
        if np.linalg.norm(grad_cur, ord=2) < precision:
            break
        cur_x = cur_x - grad_cur * learning_rate
        print(f"第{i}次迭代，x值为: {cur_x}")
    print("局部最小值x：", cur_x)
    return cur_x

if __name__ == '__main__':
    gradient_descent_2d(gradience_2d, cur_x=np.array([1, -1]), learning_rate=0.2, precision=0.000001, max_iters=10000)