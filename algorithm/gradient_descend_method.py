import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

'''
#y=2 * (x1) + (x2) + 3
rate = 0.001
x_train = np.array([[1, 2],[2, 1],[2, 3],[3, 5],[1, 3],[4, 2],[7, 3],[4, 5],[11, 3],[8, 7]])
y_train = np.array([7, 8, 10, 14, 8, 13, 20, 16, 28, 26])
x_test  = np.array([[1, 4],[2, 2],[2, 5],[5, 3],[1, 5],[4, 1]])


a = np.random.normal()
b = np.random.normal()
c = np.random.normal()


#线性方程
def h(x):
    return a*x[0]+b*x[1]+c


for i in range(1000):
    sum_a=0
    sum_b=0
    sum_c=0
    for x, y in zip(x_train, y_train):
        sum_a = sum_a + rate*(y-h(x))*x[0]
        sum_b = sum_b + rate*(y-h(x))*x[1]
        sum_c = sum_c + rate*(y-h(x))
    a = a + sum_a
    b = b + sum_b
    c = c + sum_c
    plt.plot([h(xi) for xi in x_test])


print(a)
print(b)
print(c)

result=[h(xi) for xi in x_train]
print(result)

result=[h(xi) for xi in x_test]
print(result)

plt.show()

'''

"""
'''批量梯度下降法'''
#alpha 和 epsilon 两个参数选取时，需要稍加注意，可能会导致函数无法收敛的情况

# 训练集
# 每个样本点有3个分量 (x0,x1,x2)
x = [(1, 0., 3), (1, 1., 3), (1, 2., 3), (1, 3., 2), (1, 4., 4)]
# y[i] 样本点对应的输出
y = [95.364, 97.217205, 75.195834, 60.105519, 49.342380]

# 迭代阀值，当两次迭代损失函数之差小于该阀值时停止迭代
epsilon = 0.001

# 学习率
alpha = 0.01  #学习步长
diff = [0, 0]  #残差存放参数
max_itor = 1000
error1 = 0  #残差平方和存放参数
error0 = 0  #最小残差平方和存放参数
cnt = 0
m = len(x)

# 初始化参数
theta0 = 0
theta1 = 0
theta2 = 0

while True:
    cnt += 1

    # 参数迭代计算
    for i in range(m):
        # 拟合函数为 y = theta0 * x[0] + theta1 * x[1] +theta2 * x[2]
        # 计算残差
        diff[0] = (theta0 + theta1 * x[i][1] + theta2 * x[i][2]) - y[i]

        # 梯度 = diff[0] * x[i][j]
        theta0 -= alpha * diff[0] * x[i][0]
        theta1 -= alpha * diff[0] * x[i][1]
        theta2 -= alpha * diff[0] * x[i][2]

    # 计算损失函数
    error1 = 0
    #计算二分之一残差平方和
    for lp in range(len(x)):
        error1 += (y[lp] - (theta0 + theta1 * x[lp][1] + theta2 * x[lp][2])) ** 2 / 2

    if abs(error1 - error0) < epsilon:
        break
    else:
        error0 = error1

    #print('theta0:%f,theta1:%f,theta2:%f,error1:%f'%(theta0,theta1,theta2,error1))

print ('Done: theta0 : %f, theta1 : %f, theta2 : %f' % (theta0, theta1, theta2))
print('迭代次数: %d' % cnt)
"""

####################


# 构造训练数据
x = np.arange(0., 10., 0.2)
m = len(x)  # 训练数据点数目
x0 = np.full(m, 1.0)
input_data = np.vstack([x0, x]).T  # 将偏置b作为权向量的第一个分量
target_data = 2 * x + 5 + np.random.randn(m)

# 两种终止条件
loop_max = 10000  # 最大迭代次数(防止死循环)
epsilon = 1e-3

# 初始化权值
np.random.seed(0)
w = np.random.randn(2)

# w = np.zeros(2)

alpha = 0.001  # 步长(注意取值过大会导致振荡,过小收敛速度变慢)
diff = 0.
error = np.zeros(2)
count = 0  # 循环次数
finish = 0  # 终止标志

# -------------------------------------------随机梯度下降算法----------------------------------------------------------
'''
while count < loop_max:
    count += 1

    # 遍历训练数据集，不断更新权值
    for i in range(m):  
        diff = np.dot(w, input_data[i]) - target_data[i]  # 训练集代入,计算误差值

        # 采用随机梯度下降算法,更新一次权值只使用一组训练数据
        w = w - alpha * diff * input_data[i]

        # ------------------------------终止条件判断-----------------------------------------
        # 若没终止，则继续读取样本进行处理，如果所有样本都读取完毕了,则循环重新从头开始读取样本进行处理。

    # ----------------------------------终止条件判断-----------------------------------------
    # 注意：有多种迭代终止条件，和判断语句的位置。终止判断可以放在权值向量更新一次后,也可以放在更新m次后。
    if np.linalg.norm(w - error) < epsilon:     # 终止条件：前后两次计算出的权向量的绝对误差充分小  
        finish = 1
        break
    else:
        error = w
print 'loop count = %d' % count,  '\tw:[%f, %f]' % (w[0], w[1])
'''

# -----------------------------------------------梯度下降法-----------------------------------------------------------

while count < loop_max:
    count += 1

    # 标准梯度下降是在权值更新前对所有样例汇总误差，而随机梯度下降的权值是通过考查某个训练样例来更新的
    # 在标准梯度下降中，权值更新的每一步对多个样例求和，需要更多的计算
    sum_m = np.zeros(2)
    for i in range(m):
        dif = (np.dot(w, input_data[i]) - target_data[i]) * input_data[i]
        sum_m = sum_m + dif  # 当alpha取值过大时,sum_m会在迭代过程中会溢出

    w = w - alpha * sum_m  # 注意步长alpha的取值,过大会导致振荡
    # w = w - 0.005 * sum_m      # alpha取0.005时产生振荡,需要将alpha调小

    # 判断是否已收敛
    if np.linalg.norm(w - error) < epsilon:
        finish = 1
        break
    else:
        error = w

print ('loop count = %d' % count, '\tw:[%f, %f]' % (w[0], w[1]))


# check with scipy linear regression
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x, target_data)
print ('intercept = %s slope = %s' % (intercept, slope))


'''
plt.plot(x, target_data, 'k+')
plt.plot(x, w[1] * x + w[0], 'r')
plt.show()
'''