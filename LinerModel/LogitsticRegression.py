import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class GradientDescent():
    def __init__(self,n):
        self.w = np.random.rand(n).astype(float)
        self.w = self.w.reshape(-1,1)
        self.lambdaL2 = 0
        return

    def forward(self,x):
        return x @ self.w

    def grad(self,y,preY,x):
        n = x.shape[0]
        g = (1 / n ) * ( x.T @ (preY - y) + self.lambdaL2 * self.w)
        return g

    def sigmoid(self,y):
        return 1 / (1 + np.exp( - y))
    def binaryCrossEntropy(self,y,preY):

        # 矩阵中对应位置相乘
        return np.mean(-1 * (y * np.log(preY ) + (1 - y) * np.log(1 - preY )))


class Solve():
    # 加一列 1 ，认为是 b * x -> b * 1
    def addOneInColumn(self, data):
        npOne = np.ones((data.shape[0], 1))
        return np.concatenate((data, npOne), axis=1)
        # return npOne


    def solve2(self,trainX,trainY,testX,testY):
        n = trainX.shape[1]
        # n为特征值个数
        gmodel = GradientDescent(n)
        learning_rate = 0.1
        numIterations = 10000
        lossHistory = []

        for epoch in range(numIterations):
            preY = gmodel.forward(trainX)
            sigmoidY = gmodel.sigmoid( preY)

            loss = gmodel.binaryCrossEntropy(trainY,sigmoidY)
            lossHistory.append(loss)
            grad = gmodel.grad(trainY,sigmoidY,trainX)
            gmodel.w = gmodel.w - learning_rate * grad

        print("求得梯度下降的参数w：")
        print(gmodel.w)
        plt.plot(range(numIterations), lossHistory)
        plt.xlabel("迭代次数")
        plt.ylabel("损失")
        plt.title("损失曲线")
        plt.show()
        #测试
        testPreY = gmodel.forward(testX)
        testSigmodY = gmodel.sigmoid(testPreY)
        sum = testSigmodY.shape[0]
        correct = 0

        for element1,element2 in zip(testSigmodY,testY):
            if element2 == 0:
                if element1 < 0.5:
                    correct = correct + 1
            if element2 == 1:
                if element1 > 0.5:
                    correct = correct + 1
        # 有些过拟合
        print("正确率{}/{}".format(correct,sum))
        testLoss = gmodel.binaryCrossEntropy(testPreY,testSigmodY)
        print("二元交叉熵损失值:{}".format(testLoss))
        return
if __name__ == '__main__':
    # 1.数据清理
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号
    solve = Solve()
    data = np.loadtxt("data2.txt", delimiter=',')
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

    # ------------------------------
    # 梯度下降法
    solve.solve2(trainX,trainY,testX,testY)
