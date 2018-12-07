#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import operator as op


# 根据欧式距离计算向量点之间的距离
# inputX 用于分类的输入向量X
# dataSet  样本集
# labels  标签，个数与样本集行数相同
# k  kNN ，计算输入向量与所有样本点的欧式距离，选出距离最近的k个特征向量，投票，选出类别
def classify0(inputX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # dataSet 行数
    
    #在列向量方向上重复inX共1次(横向)，行向量方向上重复inX共dataSetSize次(纵向)
    diffMat = np.tile(inputX, (dataSetSize, 1)) - dataSet   #复制inputX, 再跟样本集的每个行向量（特征向量）相减
    #print("inputX is", inputX)
    #print(diffMat)
    
    sqDiffMat = diffMat**2  #平方
    #print(sqDiffMat)
    
    sqDistances = sqDiffMat.sum(axis=1)  #对行向量求和； 如果axis=0,对列向量求和
    #print(sqDistances)
    
    distances = sqDistances**0.5  #平方根
    #print(distances)
    
    sortedDistIndicies = distances.argsort()   #返回排序后的索引（默认是降序）
    #print(sortedDistIndicies)
    
    classCount = {}  #字典
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]  #得到类别标签
        #字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1   #往字典里写key:value, key是类别标签，value是该类的数量
     
    #print(classCount.items())
    
    #key=operator.itemgetter(1)根据字典的值进行排序
    #key=operator.itemgetter(0)根据字典的键进行排序
    sortedClassCount = sorted(classCount.items(), key=op.itemgetter(1), reverse=True)
    
    return sortedClassCount[0][0]    


# 从文本文件中解析数据
# 返回： 特征矩阵 和 类别label向量
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines=len(arrayOLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()   #去掉空格
        listFromLine = line.split('\t')   # 分隔后存到列表里 list 
                       
        returnMat[index,:] = listFromLine[0:3]   #复制到returnMat的一行，index是第几行的意思，不是pandas的DataFrame里的index
        
        classLabelVector.append(int(listFromLine[-1]))   #把最后一列（类别值）放到标签向量里, 注意要转换数据类型，否则就是字符型
        index +=1
    return returnMat, classLabelVector
            

# 归一化特征值, 把值归一化到 0-1 之间
def autoNorm(dataSet):
    minVals = dataSet.min(0)  #对第一轴求最小，轴是依次遍历操作，比如矩阵是3*6，有两个轴，0：第一轴（列方向）；  1：第二轴(行方向)
    #print(minVals)
    
    maxVals = dataSet.max(0)  #对第一轴求最大
    #print(maxVals)
    
    ranges = maxVals - minVals
    
    #print(np.shape(dataSet))  # #shape(dataSet)返回dataSet的矩阵行数和列数 （元组）
    normDataSet = np.zeros(np.shape(dataSet))   
    
    m = dataSet.shape[0]  #返回dataSet的行数

    #原始值减去最小值
    normDataSet = dataSet - np.tile(minVals, (m,1))
    
    #除以最大和最小值的差,得到归一化数据
    normDataSet = normDataSet/np.tile(ranges, (m,1))
    
    #返回归一化数据结果,数据范围,最小值
    return normDataSet, ranges, minVals


def datingClassTest():

    # 取所有数据的百分之十
    hoRatio = 0.10

    # 将返回的特征矩阵和分类向量分别存储到datingDataMat和datingLabels中
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')

    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]  # 获得normMat的行数
    numTestVecs = int(m*hoRatio)  #百分之十作为测试数据的个数
    errorCount = 0.0

    for i in range(numTestVecs):
        # 前numTestVecs个数据作为测试集,后m-numTestVecs个数据作为训练集
        k = 3 # kNN的超参数
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m],ｋ)
        print("the classifier came back with: %d, the real answer is: %d" %(classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]) : 
            errorCount += 1.0
            print("the total error rate is：%f"%(errorCount/float(numTestVecs)))


#datingClassTest()

def classifyPerson():
    resultList = ['Not at all', 'in small doses','in large doses']

    # 三维特征用户输入
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))

    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)

    # 生成NumPy数组,测试集
    inArr = np.array([ffMiles, percentTats, iceCream])

    # 测试集归一化
    norminArr = (inArr - minVals) / ranges

    classifierResult = classify0(norminArr, normMat, datingLabels, 3)
    print("you will probably like this person: ", resultList[classifierResult - 1])


classifyPerson()







