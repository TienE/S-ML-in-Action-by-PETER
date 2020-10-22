# -*- coding: utf-8 -*-

import numpy as np
import operator

def classfy_knn(dataSet_l, k, inVection):
    """
    Parameters
    ----------
    dataSet_l : LIST
        样本数据,带有标签的.
    k : INT
        k值，对应最靠近的k个样本的类值.
    inVection : LIST
        待分类数据,不带标签的.

    Returns
    -------
    None.
    """
    # S1 -- 数据形式处理：
    # 标签，另成一组
    labels = []
    for item in dataSet_l:
        labels.extend(item[-1])
        item.pop(-1)
    # 转化成 np.array 型
    dataSet = np.array(dataSet_l,dtype= np.uint8)
 
    # 获取 数组~长度：样本个数
    dataSetSize = dataSet.shape[0]
    # 获得 用于计算~矩阵数组。
    inVSet = np.array(np.tile(inVection, (dataSetSize, 1)))
    

    # S2 -- 计算 -- 算子：欧氏距离
    diffMat = dataSet - inVSet
    sqDiffmat = diffMat**2
    sqDistance = sqDiffmat.sum(axis = 1)
    distance = sqDistance**(0.5)
    
    
    # S3 -- 对‘距离’进行排序，得到k个近值；得到字典 -- 记录k值含有的类别，和对应的个数
    # 对‘距离’进行排序，并返回‘下标值’
    sortDistanceIndex = distance.argsort()
    # 寻找 k 个最进~类，并计数。
    # 用于计数
    classCount = {}
    for i in range(k):
        # 用 第 i 近样本~下标 返回‘类别标签’
        vote_label = labels[sortDistanceIndex[i]]
        # 记录 此类‘标签’有多少个：
        classCount[vote_label] = classCount.get(vote_label, 0) + 1
    

    # S4 -- 对字典排序，选出字典中“个数最多的类别”
    # 排序：
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1), reverse = True)
    # 得到类别：
    return sortedClassCount[0][0]
    
