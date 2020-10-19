def func_1d(x):
    """
    目标函数：y = x ** 2 + 1
    param x: 自变量
    return : 因变量
    """
    return x ** 2 + 1

def gradience_1d(x):
    """
    求导后的目标函数梯度：r = 2 * x
    """
    return x * 2

def gradient_descent_1d(gradience, cur_x=0.1, learning_rate=0.01, precision=0.0001, max_iters=10000):
    """
    一维梯度下降：
    param gradience:目标函数梯度
    param init_x:提供的初始值
    param learning_rate:学习速率
    param precision:收敛精度
    param max_iters:最大迭代次数
    """
    for i in range(max_iters):
        grad_cur = gradience(cur_x)
        if abs(grad_cur) < precision:
            break
        cur_x = cur_x - learning_rate * grad_cur
        print(f"第{i}次迭代，x值为：{cur_x}")
    print("局部最小值：x = ", cur_x)
    return cur_x

if __name__ == '__main__':
    gradient_descent_1d(gradience_1d, cur_x=5, learning_rate=0.2, precision=0.000001, max_iters=10000)
