import matplotlib
import numpy

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.metrics import accuracy_score


class SVMLinear:
    def __init__(self):
        self.pipeline = Pipeline([
            ('scaler',StandardScaler()),
            ('svm',SVC(kernel = 'linear',C = 1e18))
        ])
    def scalerData(self,x):
        return self.pipeline['scaler'].fit_transform(x)

    def testScalerData(self,x):
        return self.pipeline['scaler'].transform(x)

    def fit(self,x,y):
        self.pipeline['svm'].fit(x,y)
    #
    def predict(self,x,y):
        predY = self.pipeline.predict(x)
        accuracy = accuracy_score(y,predY)
        return predY,accuracy

    def getParameter(self):
        svmClf = self.pipeline.named_steps['svm']
        w = svmClf.coef_[0]
        b = svmClf.intercept_
        return w,b

def drawChart(w,b,trainX,trainY,testX,testY):
    print((testX))
    print(testY)

    xRed = []
    xBlue = []

    n = trainX.shape[0]
    for i in range(0,n):
        if(trainY[i] == 1):
            xRed.append(trainX[i])
        else:
            xBlue.append(trainX[i])
    redArray = numpy.array(xRed)
    blueArray = numpy.array(xBlue)
    plt.scatter(redArray[:,0],redArray[:,1],color = 'red', label = '训练数据正值')
    plt.scatter(blueArray[:,0],blueArray[:,1],color= 'blue',label = '训练数据负值')


    xPurple = []
    xGreen = []
    n = testX.shape[0]
    for i in range(0, n):
        if (testY[i] == 1):
            xPurple.append(testX[i])
        else:
            xGreen.append(testX[i])
    purpleArray = numpy.array(xPurple)
    GreenArray = numpy.array(xGreen)
    plt.scatter(purpleArray[:, 0], purpleArray[:, 1], color='purple',label = '测试数据正值')
    plt.scatter(GreenArray[:, 0], GreenArray[:, 1], color='green',label = '测试数据负值')

    xValues = np.linspace(-3,3,100)
    yValues =  -1 * (xValues * w[0] + b ) / w[1]


    upY = - (w[0] * xValues + b - 1) / w[1]
    downY = - (w[0] * xValues + b + 1) / w[1]

    plt.plot(xValues,yValues,color = 'black',)
    plt.plot(xValues,upY,color = 'black',linestyle = '--')
    plt.plot(xValues,downY,color = 'black',linestyle = '--')



    plt.legend()


    plt.show()
    return

if __name__ == '__main__':
    # 读取数据
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号
    data = loadmat("data1.mat")
    # print(data)
    dataX = data['X']
    dataY = data['y']
    splitIndex = int(len(dataX) * 0.7)

    trainX,testX = np.split(dataX,[splitIndex])
    trainY,testY = np.split(dataY,[splitIndex])

    svmLinear = SVMLinear()
    trainX = svmLinear.scalerData(trainX)
    testX = svmLinear.testScalerData(testX)

    svmLinear.fit(trainX,trainY)

    ans,accuracy = svmLinear.predict(testX,testY)

    w,b = svmLinear.getParameter()

    drawChart(w,b,trainX,trainY,testX,testY)

    print("测试数据得正确率：{}".format(accuracy * 100 ))

