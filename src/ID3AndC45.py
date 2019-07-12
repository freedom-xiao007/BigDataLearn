#!/usr/bin/env python
# @Time    : 2019/7/11 17:04
# @Author  : LiuWei
# @Site    : 
# @File    : ID3AndC45.py
# @Software: PyCharm
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
from copy import copy

import math


def calculateInfoEntropy(trainData, attributeIndex):
    """

    :param trainData:
    :param attributeIndex:
    :return:
    {
        status: {
            count: value,
            value: value,
        },
        ......
    }
    """
    statusStatistics = {}
    for item in trainData:
        if item[attributeIndex] not in statusStatistics:
            statusStatistics[item[attributeIndex]] = {}
        if trainData[item] not in statusStatistics[item[attributeIndex]]:
            statusStatistics[item[attributeIndex]][trainData[item]] = 1
        else:
            statusStatistics[item[attributeIndex]][trainData[item]] = statusStatistics[item[attributeIndex]][trainData[item]] + 1

    infoEntropyMap = {}
    for status in statusStatistics:
        amount = 0
        for result in statusStatistics[status]:
            amount = amount + statusStatistics[status][result]

        infoEntropy = 0.0
        for result in statusStatistics[status]:
            probability = statusStatistics[status][result] / amount
            infoEntropy = infoEntropy + (probability * math.log(probability, 2))

        infoEntropyMap[status] = {}
        infoEntropyMap[status]["count"] = amount
        infoEntropyMap[status]["value"] = infoEntropy * -1
    return infoEntropyMap


def getInfoEntropy(trainData, properties, propertiesIndex):
    """

    :param trainData:
    :param properties:
    :param propertiesIndex:
    :return:
    {
        attribute: {
            status: {
                count: value,
                value: value,
            },
            ......
        },
        ......
    }
    """
    infoEntropy = {}
    for attribute in properties:
        infoEntropy[attribute] = calculateInfoEntropy(trainData, propertiesIndex[attribute])
    return infoEntropy


def calculateNormalizedInfoEntropy(attributeInfoEntropy):
    amount = 0
    for status in attributeInfoEntropy:
        amount = amount + attributeInfoEntropy[status]["count"]

    normalizedInfoEntropy = 0.0
    for status in attributeInfoEntropy:
        probability = attributeInfoEntropy[status]["count"] / amount
        normalizedInfoEntropy = normalizedInfoEntropy + probability * attributeInfoEntropy[status]["value"]
    return normalizedInfoEntropy


def getNormalizedInfoEntropy(infoEntropy):
    normalizedInfoEntropy = {}
    for attribute in infoEntropy:
        normalizedInfoEntropy[attribute] = calculateNormalizedInfoEntropy(infoEntropy[attribute])
    return normalizedInfoEntropy


def getAttributeEntropy(trainData, properties, propertiesIndex):
    attributeEntropy = {}
    for attribute in properties:
        attributeEntropy[attribute] = 0.0
        index = propertiesIndex[attribute]
        attributeStatistics = {}
        for item in trainData:
            if item[index] not in attributeStatistics:
                attributeStatistics[item[index]] = 1
            else:
                attributeStatistics[item[index]] = attributeStatistics[item[index]] + 1

        for item in attributeStatistics:
            probability = attributeStatistics[item] / len(trainData)
            attributeEntropy[attribute] = attributeEntropy[attribute] + probability * math.log(probability, 2)
        attributeEntropy[attribute] = attributeEntropy[attribute] * -1
    return attributeEntropy


def getInfoGainRate(infoGain, attributeEntropy):
    infoGainRate = {}
    for attribute in attributeEntropy:
        if attributeEntropy[attribute] == 0.0:
            continue
        infoGainRate[attribute] = infoGain[attribute] / attributeEntropy[attribute]
    return infoGainRate


def selectNode(trainData, properties, propertiesIndex, model):
    """

    :param trainData:
    :param properties:
    :param propertiesIndex:
    :return:
    node(当前选择的节点属性信息):{
        name: 节点属性名称
        status: { 各个状态的信息熵
            key: {
                keys: []
                value: value
            }
        }
    }
    """
    print("\n当前训练集为:", trainData)
    print("当前的属性列表为:", properties)
    resultStatistics = {}
    for item in trainData:
        if trainData[item] not in resultStatistics:
            resultStatistics[trainData[item]] = 1
        else:
            resultStatistics[trainData[item]] = resultStatistics[trainData[item]] + 1

    rootInfoEntropy = 0.0
    for result in resultStatistics:
        probability = resultStatistics[result] / len(trainData)
        rootInfoEntropy = rootInfoEntropy + (probability * math.log(probability, 2))
    rootInfoEntropy = rootInfoEntropy * -1
    print("当前训练集的根节点的信息熵:", rootInfoEntropy)

    infoEntropy = getInfoEntropy(trainData, properties, propertiesIndex)
    print("当前训练集的信息熵:", infoEntropy)

    normalizedInfoEntropy = getNormalizedInfoEntropy(infoEntropy)
    print("当前训练集的归一化信息熵:", normalizedInfoEntropy)

    infoGain = {}
    for attribute in properties:
        infoGain[attribute] = rootInfoEntropy - normalizedInfoEntropy[attribute]
    print("当前训练集的信息增益:", infoGain)

    name = None
    for attribute in infoGain:
        if name is None:
            name = attribute
        elif infoGain[attribute] > infoGain[name]:
            name = attribute

    # C4.5添加start
    attributeEntropy = {}
    infoGainRate = {}
    if model == 1:
        attributeEntropy = getAttributeEntropy(trainData, properties, propertiesIndex)
        print("当前属性熵：", attributeEntropy)

        infoGainRate = getInfoGainRate(infoGain, attributeEntropy)
        print("当前信息增益率:", infoGainRate)

        name = None
        for attribute in infoGainRate:
            if name is None:
                name = attribute
            elif infoGainRate[attribute] > infoGainRate[name]:
                name = attribute

    statusMap = {}
    for status in infoEntropy[name]:
        statusMap[status] = {}
        statusMap[status]["value"] = infoEntropy[name][status]["value"]
        statusMap[status]["keys"] = []
        for item in trainData:
            if item[propertiesIndex[name]] == status:
                statusMap[status]["keys"].append(item)

    node = {
        "name": name,
        "status": statusMap,
    }
    return node


def getID3Tree(trainData, properties, propertiesIndex, model):
    """
    获取ID3决策树
    :param trainData:
    :param properties:
    :param propertiesIndex:
    :return:

    node(当前选择的节点属性信息):{
        name: 节点属性名称
        status: { 各个状态的信息熵
            key: {
                keys: []
                value: value
            }
        }
    }
    """
    node = selectNode(trainData, properties, propertiesIndex, model)
    print("当前选择的最优节点为:", node)
    tree = {
        node["name"]: {},
    }

    tempProperties = copy(properties)
    print("进行属性列表裁剪:", tempProperties, node["name"])
    if node["name"] in tempProperties:
        tempProperties.remove(node["name"])

    for status in node["status"]:
        if node["status"][status]["value"] == 0.0:
            tree[node["name"]][status] = trainData[node["status"][status]["keys"][0]]
        else:
            tempTrainData = {}
            for item in trainData:
                if item[propertiesIndex[node["name"]]] == status:
                    tempTrainData[item] = trainData[item]

            tree[node["name"]][status], nodeName = getID3Tree(tempTrainData, tempProperties, propertiesIndex, model)
            print("进行属性列表裁剪:", tempProperties, nodeName)
            if nodeName in tempProperties:
                tempProperties.remove(nodeName)

    return tree, node["name"]


def printTree(tree, interval):
    newInterval = interval + "\t"
    for item in tree:
        if type(tree[item]) is not dict:
            print(interval + item, tree[item])
        else:
            print(interval + item)
            printTree(tree[item], newInterval)


if __name__ == "__main__":
    # 设置算法模式: 0->ID3 1->C4.5
    model = 0

    properties = ["weather", "temperature", "humidity", "windy"]
    propertiesIndex = {"weather": 0, "temperature": 1, "humidity": 2, "windy": 3}
    # 0:不打篮球 1:打篮球
    trainData = {
        ("sun", "high", "middle", "no"): 0,
        ("sun", "high", "middle", "yes"): 0,
        ("cloud", "high", "high", "no"): 1,
        ("rain", "high", "high", "no"): 1,
        ("rain", "low", "high", "no"): 0,
        ("sun", "middle", "middle", "yes"): 1,
        ("cloud", "middle", "high", "yes"): 0,
    }
    # for key in trainData:
    #     print(key, trainData[key])

    tree, name = getID3Tree(trainData, properties, propertiesIndex, model)
    print(str(tree).replace("'", '"'))
    printTree(tree, "")
