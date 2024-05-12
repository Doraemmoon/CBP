import torch
import numpy as np
import math
from utils import generateBatchSamples
import torch.nn as nn


def evaluate_ranking(model, dataLoader, config, device, isTrain):

    evalBatchSize = config.batch_size

    if isTrain:  # Train(test)
        numUser = dataLoader.numValid
        numItems = dataLoader.numItemsTrain
    else:  # Test(test)
        numUser = dataLoader.numTest
        numItems = dataLoader.numItemsTest

    if numUser % config.batch_size == 0:
        numBatch = numUser // evalBatchSize
    else:
        numBatch = numUser // evalBatchSize + 1

    idxList = [i for i in range(numUser)]

    Recall = []
    NDCG = []

    for batch in range(numBatch):
        start = batch * evalBatchSize
        end = min(batch * evalBatchSize + evalBatchSize, numUser)

        batchList = idxList[start:end]

        # target is the same with targetList in evaluation
        samples, decays, userhis, itemhis, his, target, offset, lenX, _, \
        userPOP, itemPOP = generateBatchSamples(dataLoader, batchList,
                                                config, isEval=1)

        users = torch.from_numpy(np.array(batchList)).type(torch.LongTensor).to(device)
        samples = torch.from_numpy(samples).type(torch.LongTensor).to(device)
        decays = torch.from_numpy(decays).type(torch.FloatTensor).to(device)
        userhis = torch.from_numpy(userhis).type(torch.FloatTensor).to(device)
        itemhis = torch.from_numpy(itemhis).type(torch.FloatTensor).to(device)
        his = torch.from_numpy(his).type(torch.FloatTensor).to(device)
        userPOP = torch.from_numpy(userPOP).type(torch.FloatTensor).to(device)
        itemPOP = torch.from_numpy(itemPOP).type(torch.FloatTensor).to(device)
        offset = torch.from_numpy(offset).type(torch.FloatTensor).to(device)
        lenX = lenX.to(device)

        with torch.no_grad():
            scores, _ = model.forward(users, samples, decays, offset,
                                      userhis, itemhis, his, userPOP, itemPOP, device)

        # get the index of top 40 items
        predIdx = torch.topk(scores, 40, largest=True)[1]
        predIdx = predIdx.cpu().data.numpy().copy()

        if batch == 0:
            predIdxArray = predIdx
            targetList = target
            # if config.model == 'SNBR' and config.abla=="None":
            #     gateArray = gate
        else:
            predIdxArray = np.append(predIdxArray, predIdx, axis=0)
            targetList += target

    for k in [5, 10, 20]:
        recall = calRecall(targetList, predIdxArray, k)
        Recall.append(recall)
        NDCG.append(calNDCG(targetList, predIdxArray, k))
    return Recall, NDCG


def calRecall(target, pred, k):
    assert len(target) == len(pred)
    sumRecall = 0
    for i in range(len(target)):
        gt = set(target[i])
        ptar = set(pred[i][:k])

        if len(gt) == 0:
            print('Error')

        sumRecall += len(gt & ptar) / float(len(gt))

    recall = sumRecall / float(len(target))

    return recall


def calNDCG(target, pred, k):
    assert len(target) == len(pred)
    sumNDCG = 0
    for i in range(len(target)):
        valK = min(k, len(target[i]))
        gt = set(target[i])
        idcg = calIDCG(valK)
        dcg = sum([int(pred[i][j] in gt) / math.log(j + 2, 2) for j in range(k)])
        sumNDCG += dcg / idcg

    return sumNDCG / float(len(target))


# the gain is 1 for every hit, and 0 otherwise
def calIDCG(k):
    return sum([1.0 / math.log(i + 2, 2) for i in range(k)])


def calHr(target, pred, k):
    assert len(target) == len(pred)

    for i in range(len(target)):
        tar = set(pred[i][:k])
        if i in tar:
            return 1
    return 0
