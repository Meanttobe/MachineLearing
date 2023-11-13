import numpy as np
import sklearn.svm
from matplotlib import pyplot as plt
from scipy.io import loadmat
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import warnings
# action参数可以设置为ignore，一位一次也不喜爱你是，once表示为只显示一次
warnings.filterwarnings(action='ignore')

class SVMNonLinear:
    def __init__(self,gama,c):
        self.pipeline = Pipeline([
            ('scaler',StandardScaler()),
            ('svm',SVC(C=c,kernel='rbf',gamma=gama,decision_function_shape='ovo'))
        ])

    def scaler(self,x):
        return self.pipeline['scaler'].fit_transform(x)
    def testScalerData(self, x):
        return self.pipeline['scaler'].transform(x)
    def fit(self,x,y):
        self.pipeline['svm'].fit(x,y)


    def predict(self,x,y):
        predY = self.pipeline['svm'].predict(x)
        accuracy = accuracy_score(predY,y)
        return accuracy



def drawChart(svm,trainX,trainY,testX,testY,g,c):


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
    purpleArray = np.array(xPurple)
    GreenArray = np.array(xGreen)
    plt.scatter(purpleArray[:, 0], purpleArray[:, 1], color='purple',label = '测试数据正值')
    plt.scatter(GreenArray[:, 0], GreenArray[:, 1], color='green',label = '测试数据负值')

    xMin, xMax = trainX[:, 0].min() - 1, trainX[:, 0].max() + 1
    yMin, yMax = trainX[:, 1].min() - 1, trainX[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(xMin, xMax, 0.01),
                         np.arange(yMin, yMax, 0.01))

    z = svm.pipeline['svm'].predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=plt.cm.brg, alpha=0.2)
    plt.title('gama = {}, C = {}'.format(g,c))
    plt.legend()
    plt.show()
    return

if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号
    data = loadmat('data2.mat')
    dataX = data['X']
    dataY = data['y']
    trainX,testX,trainY,testY=train_test_split(dataX,dataY,random_state=1,train_size=0.7,test_size=0.3)
    # gama = [0.1,1,10,100,1000]
    gama = [1000,100,10,1,0.1]
    C = [1000,100,10,1,0.1]
    for i in range(0,5):
        for j in range(0,5):
            svm = SVMNonLinear(gama[i],C[j])
            trainX = svm.scaler(trainX)
            testX = svm.testScalerData(testX)

            svm.fit(trainX,trainY)
            accuracy = svm.predict(testX,testY)


            drawChart(svm,trainX,trainY,testX,testY,gama[i],C[j])

            print("测试数据得正确率：{}".format(accuracy * 100))






