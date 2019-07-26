#!/usr/bin/env python
# @Time    : 2019/7/26 14:20
# @Author  : Freedom
# @Site    : 
# @File    : DecisionTreeCreate.py
# @Software: PyCharm
# -*- coding: utf-8 -*-
"""决策树构造生成ID3"""
import math


def calculateInformationGain(trainingData, rootInformationEntropy, statistics, amount):
    """计算当前属性的信息增益
    1.统计各个选项的结果情况
    2.计算各个选项的归一化信息熵
    3.计算并返回信息增益

    statistics = {
        option: {
            选项导致的各种结果的数量
            result1: int,
            result2: int,
            ......,
        },
        ......,
    }

    amount = {
        option: int,各个选项的数量
        ......,
    }
    """
    informationGain = 0.0
    dataAmount = len(trainingData)
    for option in statistics:
        optionCount = amount[option]
        informationEntropy = calculateInformationEntropy(statistics[option], optionCount)
        informationGain = informationGain + optionCount / dataAmount * informationEntropy
    print(rootInformationEntropy, informationGain, statistics)
    return rootInformationEntropy - informationGain


def getResultStatistics(trainingData):
    """对数据集的结果进行统计，用于信息熵计算"""
    statistics = {}
    for row in trainingData:
        result = row[-1]
        if result not in statistics:
            statistics[result] = 0
        statistics[result] = statistics[result] + 1
    return statistics


def calculateInformationEntropy(resultStatistics, amount):
    """信息熵计算"""
    result = 0.0
    for value in resultStatistics.values():
        probability = value / amount
        result = result + probability * math.log(probability, 2)
    return result * -1


def getAllStatistics(attributes, attributeIndex, trainingData):
    """对表格中的数据进行统计，得到每个属性对应的选项所导致的结果的统计表和各个选项的数量
    statistics = {
        attribute: { 属性
            option: { 选项
                选项导致的各种结果的数量
                result1: int,
                result2: int,
                ......,
            },
            ......,
        },
        ......,
    }

    amount = {
        attribute: {
            option: int,各个选项的数量
            ......,
        }
    }
    """
    statistics = {}
    amount = {}
    for row in trainingData:
        value = row[-1]
        for attribute in attributes:
            if attribute not in statistics:
                statistics[attribute] = {}
            if attribute not in amount:
                amount[attribute] = {}

            option = row[attributeIndex[attribute]]
            if option not in statistics[attribute]:
                statistics[attribute][option] = {}
            if value not in statistics[attribute][option]:
                statistics[attribute][option][value] = 0
            statistics[attribute][option][value] = statistics[attribute][option][value] + 1

            if option not in amount[attribute]:
                amount[attribute][option] = 0
            amount[attribute][option] = amount[attribute][option] + 1
    return statistics, amount


def selectOptimalProperties(attributes, attributeIndex, trainingData):
    """最优节点选择
    1.计算各个属性的信息增益
    2.选择信息增益率最大的进行返回

    statistics = {
        attribute: { 属性
            option: { 选项
                选项导致的各种结果的数量
                result1: int,
                result2: int,
                ......,
            },
            ......,
        },
        ......,
    }

    amount = {
        attribute: {
            option: int,各个选项的数量
            ......,
        }
    }
    """
    statistics, amount = getAllStatistics(attributes, attributeIndex, trainingData)
    values = {}
    for attribute in attributes:
        resultStatistics = getResultStatistics(trainingData)
        rootInformationEntropy = calculateInformationEntropy(resultStatistics, len(trainingData))
        values[attribute] = calculateInformationGain(trainingData, rootInformationEntropy, statistics[attribute],
                                                     amount[attribute])
    print("information gain:", values)
    name = max(values, key=values.get)
    return name, statistics[name]


def deletePropertiesIndex(attributeIndex, name):
    """删除已经选择的属性，生成新的属性值与数据集的位置映射"""
    newPropertiesIndex = {}
    index = 0
    for attribute in attributeIndex.keys():
        if attribute == name:
            continue
        newPropertiesIndex[attribute] = index
        index = index + 1
    return newPropertiesIndex


def deleteTrainData(option, attributeIndex, trainingData):
    """选择当前属性选项的相关数据，生成新的数据集"""
    newTrainingData = []
    for row in trainingData:
        if row[attributeIndex] == option:
            newTrainingData.append(row[:attributeIndex] + row[attributeIndex + 1:])
    return newTrainingData


def createTree(attributes, attributeIndex, trainingData):
    """决策树生成
    1.选择节点最优属性
    2.判断当前属性的选择是否能确定唯一结果，不能则继续向下分裂
    """
    tree = {}
    name, options = selectOptimalProperties(attributes, attributeIndex, trainingData)
    tree[name] = {}
    print("select the optimal attribute:", name, options)

    for option in options:
        # 如果此选项能导致唯一的结果，不在分类，存在两种结果则继续分裂
        if len(options[option]) == 1:
            print(name, option, "is only", list(options[option].keys())[0])
            tree[name][option] = list(options[option].keys())[0]
        else:
            index = attributeIndex[name]
            newAttributes = attributes.copy()
            newAttributes.remove(name)
            newAttributeIndex = deletePropertiesIndex(attributeIndex, name)
            newTrainingData = deleteTrainData(option, index, trainingData)
            print("sub select", name, option, newAttributes, newTrainingData)
            tree[name][option], subName = createTree(newAttributes, newAttributeIndex, newTrainingData)
    return tree


def printTree(tree, interval):
    newInterval = interval + "\t"
    for item in tree:
        if type(tree[item]) is not dict:
            print(interval + item, tree[item])
        else:
            print(interval + item)
            printTree(tree[item], newInterval)


if __name__ == "__main__":
    """数据如下：
    |天气|温度|湿度|刮风|是否打篮球|
    |----|:--:|:--:|:--:|--------|
    |晴   |高  |中  |否  |否      |
    |晴   |高  |中  |是  |否      |
    |阴   |高  |高  |否  |是      |
    |小雨 |高  |高  |否  |是      |
    |小雨 |低  |高  |否  |否      |
    |晴天 |中  |中  |是  |是      |
    |阴天 |中  |高  |是  |否      |
    """
    properties = ["weather", "temperature", "humidity", "windy"]
    propertiesIndex = {"weather": 0, "temperature": 1, "humidity": 2, "windy": 3}
    # 0:不打篮球 1:打篮球
    trainData = [
        ["sun", "high", "middle", "no", "don't play"],
        ["sun", "high", "middle", "yes", "don't play"],
        ["cloud", "high", "high", "no", "play"],
        ["rain", "high", "high", "no", "play"],
        ["rain", "low", "high", "no", "don't play"],
        ["sun", "middle", "middle", "yes", "play"],
        ["cloud", "middle", "high", "yes", "don't play"],
    ]
    tree = createTree(properties, propertiesIndex, trainData)
    printTree(tree, "")