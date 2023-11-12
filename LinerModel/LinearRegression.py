import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def sub(a):
    x,y = (min(10,a.shape[0]),min(10,a.shape[1]))
    print(a[0:x,0:y])



class LinearRegression:

    def __init__(self):
        return

    def normalEquations(self,x,y):
        beta = ( np.linalg.inv(x.T @ x) ) @ x.T @ y
        return beta

    def test(self,x,w):
        return x @ w

class GradientDescent():
    def __init__(self,n):
        self.w = np.random.rand(n).astype(float)
        self.w = self.w.reshape(-1,1)

        return

    def forward(self,x):
        return x @ self.w
    def grad(self,x,y,preY):
        n = x.shape[0]
        g = (1 / n) * (x.T @ (preY - y))

        return g


class Solve():
    # 加一列 1 ，认为是 b * x -> b * 1
    def addOneInColumn(self, data):
        npOne = np.ones((data.shape[0], 1))
        return np.concatenate((data, npOne), axis=1)
        # return npOne
    def MSELose(self,preY,y):
        return np.mean(np.square(preY-y))


    def solve1(self,trainX,trainY,testX,testY):
        model = LinearRegression()
        w = model.normalEquations(trainX, trainY)
        w = np.array(w)
        w = w.reshape(-1, 1)
        # 测试
        preY = model.test(testX, w)
        # print(testY)
        loss = self.MSELose(preY, testY)
        print("正规方程所求参数：")
        print(w)
        print("正规方程测试集的MSEloss：{}".format(loss))

        return

    def solve2(self,trainX,trainY,testX,testY):
        n = trainX.shape[1]
        gmodel = GradientDescent(n)
        learning_rate = 0.01
        numIterations = 10000
        lossHistory = []

        for epoch in range(numIterations):
            preY = gmodel.forward(trainX)
            loss = self.MSELose(preY,trainY)
            lossHistory.append(loss)
            grad = gmodel.grad(trainX,trainY,preY)
            gmodel.w = gmodel.w - learning_rate * grad

        plt.plot(range(numIterations), lossHistory)
        plt.xlabel("迭代次数")
        plt.ylabel("损失")
        plt.title("损失曲线")
        plt.show()

        testPreY = gmodel.forward(testX)


        testLoss = self.MSELose(testPreY,testY)
        print("梯度下降所求参数：")
        print(gmodel.w)
        print("梯度下降测试集的MSEloss：{}".format(testLoss))


        plt.scatter(testX[:, 0], testPreY, c='r')
        plt.scatter(testX[:, 0], testY, c='b')
        # 添加标题和标签
        plt.title('示例散点图')
        plt.xlabel('X轴')
        plt.ylabel('Y轴')

        plt.show()
        plt.scatter(testX[:, 1], testPreY, c='r')
        plt.scatter(testX[:, 1], testY, c='b')
        plt.title('示例散点图')
        plt.xlabel('X轴')
        plt.ylabel('Y轴')

        plt.show()
        return



if __name__ == '__main__':
    # 1.数据清理

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号
    solve = Solve()
    data = np.loadtxt("data1.txt", delimiter=',')
    data = data.astype('float32')
    trainData, testData = train_test_split(data, test_size=0.3, random_state=42)

    dataNorm = data.copy()
    maximum = np.max(dataNorm, axis=0, keepdims=True)
    minimun = np.min(dataNorm, axis=0, keepdims=True)

    trainData = (trainData - minimun) / (maximum - minimun)
    testData = (testData - minimun) / (maximum - minimun)

    # 整理样本数据
    trainX = trainData[:, 0:2]
    trainX = solve.addOneInColumn(trainX)
    trainY = trainData[:, 2]
    trainY = trainY.reshape(-1,1)

    testX = testData[:, 0:2]
    testX = solve.addOneInColumn(testX)
    testY = testData[:, 2]
    testY = testY.reshape(-1,1)


    #正规方程求参数，构建模型
    solve.solve1(trainX,trainY,testX,testY)

    # ----------
    # 梯度下降法
    solve.solve2(trainX,trainY,testX,testY)














