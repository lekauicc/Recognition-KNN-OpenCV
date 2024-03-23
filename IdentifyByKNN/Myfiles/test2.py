# coding=utf-8
import openface
import cv2
from  forModelTest import *

#我将基础集的人脸防在同目录的BaseImages里面，测试集的放在testImages里面,测试集里面
#的图片就不需要按照文件夹放了，就是做识别用的嘛。

def kNNmodelTest(baseData, Baselabels, testData, y_true):
    labelsPredict = []
    for (index, testSample) in enumerate(testData):
        y_hat = kNNClassify(testSample, baseData, Baselabels,k=3)  # kNN算法预测的标签
        labelsPredict.append(y_hat)
        if y_hat==y_true[index]:
            flag='正确'
        else:
            flag = 'sorry,错误--!'
        print '测试输入为', y_true[index],'------>','预测结果：', y_hat, '\t\t',flag

    resCompare = [True if labelsPredict[i] == y_true[i] else False for i in range(len(y_true))]  # 比较真伪
    accuracy = sum(resCompare)*100/float(len(y_true))    # 预测的准确率
    print 'K近邻算法分类器  accuracy={:.2f}%'.format(accuracy)
    return None

if __name__ == '__main__':
    dataSet, labels = baseImageRep('BaseImages')   # 提取基础集的特征点和标签
    dataTest, labelsTest = testImagesRep('testImages')  # 提取测试集的特征点和标签
    kNNmodelTest(baseData=dataSet, Baselabels=labels, testData=dataTest, y_true=labelsTest)  # 开始测试