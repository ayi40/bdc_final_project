import sys
import numpy as np
from sklearn.datasets import load_iris
import torch

class NbcModel:
    def __init__(self, features_type, classnum):
        self.features_num = len(features_type)
        self.classnum = classnum

        self.cont_list = []
        self.dis_list = []
        for i in range(self.features_num):
            if features_type[i] == 'cont':
                self.cont_list.append(i)
            elif features_type[i] == 'dis':
                self.dis_list.append(i)
            else:
                print('There is a wrong feature type.')
                sys.exit()

        self.cls_distribiton = np.zeros(self.classnum)
        if self.cont_list:
            self.means = np.zeros((self.classnum, len(self.cont_list)))
            self.stds = np.zeros((self.classnum, len(self.cont_list)))
        self.dis = np.array([])
        self.dis_dic = []

    def fit(self, train_datas):
        if self.dis_list:
            self.maxcls = max([len(set(train_datas.T[i])) for i in self.dis_list])
            self.dis = torch.zeros((len(self.dis_list), self.maxcls, self.classnum))
            for i in range(len(self.dis_list)):
                self.common_discrete_dis(train_datas.T[self.dis_list[i]], train_datas.T[-1], i)

        for i in range(self.classnum):
            self.cls_distribiton[i] = np.sum((train_datas.T[-1] == i) == 1)/len(train_datas.T[-1])
        if self.cont_list:
            for i in range(len(self.cont_list)):
                self.common_continuous_dis(train_datas.T[self.cont_list[i]], train_datas.T[-1], i)
            self.means = np.dstack([self.means[i] for i in range(self.classnum)])
            self.stds = np.dstack([self.stds[i] for i in range(self.classnum)])

        self.means[self.means == 0] = 1e-6
        self.stds[self.stds == 0] = 1e-6
        self.dis[self.dis == 0] = 1e-6
        # print(self.means.shape, self.stds, self.dis, self.cls_distribiton)

    def predict(self, test_datas):
        y = test_datas[:, -1]
        if self.cont_list:
            X_cont = test_datas[:, self.cont_list]
            exponent = np.exp((-np.power((np.dstack([X_cont]*self.classnum)- self.means) / self.stds, 2) / 2).astype(np.float64))
            GaussProb = (1 / (np.sqrt(2 * np.pi * self.stds))) * exponent
            cont_result = torch.bmm(torch.tensor(np.log(GaussProb+1e-6).transpose([0, 2, 1])),
                           torch.tensor(np.ones((len(test_datas), len(self.cont_list), 1))))
        else:
            cont_result = np.log(torch.ones((len(test_datas), self.classnum, 1)))

        if self.dis_list:
            X_dis = self.num2index(test_datas[:, self.dis_list].T)
            X_dis = self.vec2onehot(X_dis)
            res = torch.bmm(X_dis, self.dis)
            dis_res = 1
            for i in range(len(res)):
                dis_res = dis_res* res[i]
            dis_res = np.log(dis_res[:, :, np.newaxis])
        else:
            dis_res = np.log(torch.ones((len(test_datas), self.classnum, 1)))
        return list(torch.argmax(dis_res+cont_result+np.log(self.cls_distribiton[:, np.newaxis]), 1).numpy().squeeze()), \
               list(y[:, np.newaxis].squeeze())

    def common_continuous_dis(self, datas, label, feature_index):
        for i in range(self.classnum):
            mean = np.mean(datas[label == i])
            std = np.std(datas[label == i])
            self.means[i][feature_index] = mean
            self.stds[i][feature_index] = std

    def common_discrete_dis(self, datas, label, feature_index):
        eles = list(set(datas))
        dic = {}
        for classindex,i in enumerate(eles):
            dic.update({i:classindex})
        self.dis_dic.append(dic)
        for i in range(self.classnum):
            clsdata = list(datas[label == i])
            for j in range(len(eles)):
                self.dis[feature_index][j][i] = clsdata.count(eles[j]) / len(clsdata)

    def num2index(self, data):
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j] = self.dis_dic[i][data[i][j]]
        return data

    def vec2onehot(self, dis_features):
        onehot = torch.zeros((len(self.dis_list),len(dis_features[0]),self.maxcls))
        for i in range(len(dis_features)):
            for index, j in enumerate(dis_features[i]):
                onehot[i][index][int(j)] = 1
        return onehot


