import matplotlib


matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


class SVMLinear:
    def __init__(self,C):
        self.pipeline = Pipeline([
            ('scaler',StandardScaler()),
            ('svm',SVC(kernel = 'linear',C = C))
        ])
    def scalerData(self,x):
        return self.pipeline['scaler'].fit_transform(x)

    def testScalerData(self,x):
        return self.pipeline['scaler'].transform(x)

    def fit(self,x,y):
        self.pipeline['svm'].fit(x,y)
    #
    def predict(self,x,y):
        predY = self.pipeline['svm'].predict(x)

        accuracy = accuracy_score(y,predY)
        return predY,accuracy

    def getParameter(self):
        svmClf = self.pipeline.named_steps['svm']
        w = svmClf.coef_[0]
        b = svmClf.intercept_
        return w,b

def drawChart(w,b,trainX,trainY,testX,testY,ax):
    xRed = []
    xBlue = []

    n = trainX.shape[0]
    for i in range(0,n):
        if(trainY[i] == 1):
            xRed.append(trainX[i])
        else:
            xBlue.append(trainX[i])
    redArray = np.array(xRed)
    blueArray = np.array(xBlue)
    ax.scatter(redArray[:,0],redArray[:,1],color = 'red', label = '训练数据正值')
    ax.scatter(blueArray[:,0],blueArray[:,1],color= 'blue',label = '训练数据负值')


    xPurple = []
    xGreen = []
    n = testX.shape[0]
    for i in range(0, n):
        if (testY[i] == 1):
            xPurple.append(testX[i])
        else:
            xGreen.append(testX[i])
    purpleArray = np.array(xPurple)
    GreenArray = np.array(xGreen)
    ax.scatter(purpleArray[:, 0], purpleArray[:, 1], color='purple',label = '测试数据正值')
    ax.scatter(GreenArray[:, 0], GreenArray[:, 1], color='green',label = '测试数据负值')

    xValues = np.linspace(-3,3,100)
    yValues =  -1 * (xValues * w[0] + b ) / w[1]


    upY = - (w[0] * xValues + b - 1) / w[1]
    downY = - (w[0] * xValues + b + 1) / w[1]

    ax.plot(xValues,yValues,color = 'black',)
    ax.plot(xValues,upY,color = 'black',linestyle = '--')
    ax.plot(xValues,downY,color = 'black',linestyle = '--')



    ax.legend()




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
        fig, axs = plt.subplots(3, 2, figsize=(8, 12))
        C = [0.001,0.1,1,10,1000,1e15]
        for i in range(3):
            for j in range(2):
                svmLinear = SVMLinear(C[i * 2 + j])
                ax = axs[i,j]
                trainX = svmLinear.scalerData(trainX)
                testX = svmLinear.testScalerData(testX)

                svmLinear.fit(trainX,trainY)

                ans,accuracy = svmLinear.predict(testX,testY)

                w,b = svmLinear.getParameter()

                drawChart(w,b,trainX,trainY,testX,testY,ax)
                ax.set_title('C = {}'.format(C[i * 2 + j]))
                print("测试数据得正确率：{}".format(accuracy * 100 ))
        plt.tight_layout()
        plt.show()

