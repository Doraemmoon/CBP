from scipy import sparse
import numpy as np
import pickle
from scipy.sparse import csr_matrix
import scipy
from sklearn.preprocessing import normalize
import time


class dataLoader():
    def __init__(self, config):
        root = '../data/dataset/' + config.dataset + '/' + config.dataset
        dataRoot = root + '.pkl'
        with open(dataRoot, 'rb') as f:
            dataDict = pickle.load(f)

        user2item = self.generate_user_list(dataDict)

        numRe, user2idx = self.generate_sets(user2item)
        self.numItems = self.get_num_items() + 1

        print("num_users_removed   %d" % numRe)
        print("num_valid_users   %d" % len(self.testList))
        print("num_items   %d" % self.numItems)

        # to generate his
        self.numTrain, self.numValid, self.numTrainVal, self.numTest = len(self.testList), len(self.testList), len(
            self.testList), len(self.testList)
        self.numItemsTrain, self.numItemsTest = self.numItems, self.numItems
        self.valid2train = {}
        self.test2trainVal = {}
        for i in range(len(self.trainList)):
            self.valid2train[i] = i
            self.test2trainVal[i] = i

        if config.isTrain:
            self.lenTrain = self.generateLens(self.trainList)
            self.lenVal = self.generateLens(self.validList)
        else:
            self.lenTrainVal = self.generateLens(self.trainValList)
            self.lenTest = self.generateLens(self.testList)

        start = time.time()
        if config.isTrain:

            self.userhisMatTra, self.itemhisMatTra, self.hisMatTra, self.tarMatTra, \
            self.userPOPTra, self.itemPOPTra = self.generateHis(self.trainList, isTrain=1, isEval=0)
            self.userhisMatVal, self.itemhisMatVal, self.hisMatVal, self.tarMatVal, \
            self.userPOPVal, self.itemPOPVal = self.generateHis(self.validList, isTrain=1, isEval=1)

            scipy.sparse.save_npz('his/seq/his_train/' + config.dataset + '/' + config.dataset + '_hisMatTra_' + str(
                config.testOrder) + '.npz', self.hisMatTra)
            scipy.sparse.save_npz('his/seq/his_train/' + config.dataset + '/' + config.dataset + '_hisMatVal_' + str(
                config.testOrder) + '.npz', self.hisMatVal)
            scipy.sparse.save_npz('his/seq/his_train/' + config.dataset + '/' + config.dataset + '_tarMatTra_' + str(
                config.testOrder) + '.npz', self.tarMatTra)
            with open('his/seq/his_train/' + config.dataset + '/' + config.dataset + '_tarMatVal_' + str(
                    config.testOrder) + '.pkl', 'wb') as f:
                pickle.dump(self.tarMatVal, f)
            print('Mat done')

        else:

            self.userhisMatTraVal, self.itemhisMatTraVal, self.hisMatTraVal, self.tarMatTraVal, \
            self.userPOPTraVal, self.itemPOPTraVal = self.generateHis(self.trainValList, isTrain=0, isEval=0)
            self.userhisMatTest, self.itemhisMatTest, self.hisMatTest, self.tarMatTest, \
            self.userPOPTest, self.itemPOPTest = self.generateHis(self.testList, isTrain=0, isEval=1)

            scipy.sparse.save_npz(
                'his/seq/his_test/' + config.dataset + '/' + config.dataset + '_userhisMatTraVal_' + str(
                    config.testOrder) + '.npz', self.userhisMatTraVal)
            scipy.sparse.save_npz(
                'his/seq/his_test/' + config.dataset + '/' + config.dataset + '_itemhisMatTraVal_' + str(
                    config.testOrder) + '.npz', self.itemhisMatTraVal)
            scipy.sparse.save_npz(
                'his/seq/his_test/' + config.dataset + '/' + config.dataset + '_userhisMatTest_' + str(
                    config.testOrder) + '.npz', self.userhisMatTest)
            scipy.sparse.save_npz(
                'his/seq/his_test/' + config.dataset + '/' + config.dataset + '_userhisMatTest_' + str(
                    config.testOrder) + '.npz', self.userhisMatTest)
            scipy.sparse.save_npz('his/seq/his_test/' + config.dataset + '/' + config.dataset + '_hisMatTraVal_' + str(
                config.testOrder) + '.npz', self.hisMatTraVal)
            scipy.sparse.save_npz('his/seq/his_test/' + config.dataset + '/' + config.dataset + '_hisMatTest_' + str(
                config.testOrder) + '.npz', self.hisMatTest)
            scipy.sparse.save_npz('his/seq/his_test/' + config.dataset + '/' + config.dataset + '_tarMatTraVal_' + str(
                config.testOrder) + '.npz', self.tarMatTraVal)
            with open('his/seq/his_test/' + config.dataset + '/' + config.dataset + '_tarMatTest_' + str(
                    config.testOrder) + '.pkl', 'wb') as f:
                pickle.dump(self.tarMatTest, f)
            print('Mat done')

        print("finish generating his matrix, elaspe: %.3f" % (time.time() - start))

    def generate_user_list(self, dataDict):
        all_users = list(dataDict.keys())
        user2item = {}
        for user in all_users:
            user2item[user] = dataDict.get(user, [])
        return user2item

    def generate_sets(self, user2item):
        self.trainList = []
        self.validList = []
        self.trainValList = []
        self.testList = []
        count = 0
        count_remove = 0
        user2idx = {}

        for user in user2item:
            if len(user2item[user]) < 4:
                count_remove += 1
                continue
            user2idx[user] = count
            count += 1
            self.trainList.append(user2item[user][:-2])
            self.validList.append(user2item[user][:-1])
            self.trainValList.append(user2item[user][:-1])
            self.testList.append(user2item[user])
        return count_remove, user2idx

    def get_num_items(self):
        numItem = 0
        for baskets in self.testList:
            for basket in baskets:
                for item in basket:
                    numItem = max(item, numItem)

        return numItem

    def batchLoader(self, batchIdx, isTrain, isEval):
        if isTrain and not isEval:
            train = [self.trainList[idx] for idx in batchIdx]
            trainLen = [self.lenTrain[idx] for idx in batchIdx]
            userhis = self.userhisMatTra[batchIdx, :].todense()
            itemhis = self.itemhisMatTra[batchIdx, :].todense()
            his = self.hisMatTra[batchIdx, :].todense()
            target = self.tarMatTra[batchIdx, :].todense()
            userPOP = self.userPOPTraVal[batchIdx]
            itemPOP = self.itemPOPTraVal[:]
        elif isTrain and isEval:
            train = [self.validList[idx] for idx in batchIdx]
            trainLen = [self.lenVal[idx] for idx in batchIdx]
            userhis = self.userhisMatVal[batchIdx, :].todense()
            itemhis = self.itemhisMatVal[batchIdx, :].todense()
            his = self.hisMatVal[batchIdx, :].todense()
            target = [self.tarMatVal[idx] for idx in batchIdx]
            userPOP = self.userPOPVal[batchIdx]
            itemPOP = self.itemPOPVal[:]
        elif not isTrain and not isEval:
            train = [self.trainValList[idx] for idx in batchIdx]
            trainLen = [self.lenTrainVal[idx] for idx in batchIdx]
            userhis = self.userhisMatTraVal[batchIdx, :].todense()
            itemhis = self.itemhisMatTraVal[batchIdx, :].todense()
            his = self.hisMatTraVal[batchIdx, :].todense()
            target = self.tarMatTraVal[batchIdx, :].todense()
            userPOP = self.userPOPTraVal[batchIdx]
            itemPOP = self.itemPOPTraVal[:]
        else:
            train = [self.testList[idx] for idx in batchIdx]
            trainLen = [self.lenTest[idx] for idx in batchIdx]
            userhis = self.userhisMatTest[batchIdx, :].todense()
            itemhis = self.itemhisMatTest[batchIdx, :].todense()
            his = self.hisMatTest[batchIdx, :].todense()
            target = [self.tarMatTest[idx] for idx in batchIdx]
            userPOP = self.userPOPTest[batchIdx]
            itemPOP = self.itemPOPTest[:]
        return train, trainLen, userhis, itemhis, his, target, userPOP, itemPOP

    def generateLens(self, userList):

        lens = []
        for user in userList:
            lenUser = []
            trainEUser = user[:-1]
            for bas in trainEUser:
                lenUser.append(len(bas))
            lens.append(lenUser)

        return lens

    def generateHis(self, userList, isTrain, isEval):

        if isTrain and not isEval:
            hisNUM = np.zeros((self.numTrain, self.numItemsTrain))
        elif isTrain and isEval:
            hisNUM = np.zeros((self.numValid, self.numItemsTrain))
        elif not isTrain and not isEval:
            hisNUM = np.zeros((self.numTrainVal, self.numItemsTest))
        else:
            hisNUM = np.zeros((self.numTest, self.numItemsTest))

        if not isEval and isTrain:
            tarMat = np.zeros((self.numTrain, self.numItemsTrain))
        elif not isEval and not isTrain:
            tarMat = np.zeros((self.numTrain, self.numItemsTest))
        else:
            tarMat = []

        for i in range(len(userList)):
            trainUser = userList[i][:-1]
            targetUser = userList[i][-1]

            for Bas in trainUser:
                for item in Bas:
                    hisNUM[i, item] += 1
            if not isEval:
                for item in targetUser:
                    tarMat[i, item] = 1
            else:
                tarMat.append(targetUser)

        hisMat = csr_matrix(hisNUM)
        if not isEval:
            tarMat = csr_matrix(tarMat)

        userHisMat = normalize(hisMat, norm='l1', axis=1, copy=False)
        itemHisMat = normalize(hisMat, norm='l1', axis=0, copy=False)

        userConformity = np.sum(hisNUM, axis=1)
        itemPOP = np.sum(hisNUM, axis=0)

        return userHisMat, itemHisMat, hisMat, tarMat, userConformity, itemPOP
