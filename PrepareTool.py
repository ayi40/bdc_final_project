import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import Dataset, DataLoader

class PrepareTools:
    def __init__(self):
        pass

    @staticmethod
    def cls2onehot(train_dataset, val_dataset, test_dataset, features_type, combine=True):
        #split dataset into class-type(dis) part and continue-type(cont) part
        cont_list = []
        dis_list = []
        for i in range(len(features_type)):
            if features_type[i] == 'cont':
                cont_list.append(i)
            elif features_type[i] == 'dis':
                dis_list.append(i)
            else:
                print('There is a wrong feature type.')
        train_cont, train_dis,train_y = train_dataset[:, cont_list],train_dataset[:, dis_list], train_dataset[:, -1][:, np.newaxis]
        val_cont, val_dis, val_y = val_dataset[:, cont_list], val_dataset[:, dis_list], val_dataset[:, -1][:, np.newaxis]
        test_cont, test_dis, test_y= test_dataset[:, cont_list], test_dataset[:, dis_list], test_dataset[:, -1][:, np.newaxis]
        #use training dataset to fit sklearn one-hot encoder
        enc = OneHotEncoder()
        enc.fit(train_dis)
        train_dis, val_dis, test_dis = enc.transform(train_dis).toarray(), enc.transform(val_dis).toarray(), \
                                       enc.transform(test_dis).toarray()
        if combine:
            #concate class-type(dis) part and continue-type(cont) part
            return np.hstack((train_cont, train_dis,train_y)), np.hstack((val_cont, val_dis,val_y)), np.hstack((test_cont, test_dis, test_y))
        else:
            return (train_cont, train_dis, train_y), (val_cont, val_dis, val_y), (test_cont, test_dis, test_y), enc.categories_



class CommonDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.len = len(self.dataset)

    def __getitem__(self, item):
        index = item % self.len
        input, label = self.dataset[index][:-1], self.dataset[index][-1]
        return torch.tensor(input.astype(np.float32)), torch.tensor(label)

    def __len__(self):
        return self.len


class MixDataset(Dataset):
    def __init__(self, dataset, category):
        self.cont, dis, self.y = dataset
        self.dis = []
        self.len = len(self.cont)
        self.category = category
        self.split_dis(dis)

    def __getitem__(self, item):
        input = (torch.tensor(self.cont[item].astype(np.float32)), [torch.tensor(i[item].astype(np.float32)) for i in self.dis])
        try:
            label = torch.tensor(self.y[item].astype(np.int64))
        except:
            label = torch.tensor(self.y[item].astype(np.float64))
        return input, label

    def __len__(self):
        return self.len

    def split_dis(self, dis):
        l, r = 0, 0
        for i in range(len(self.category)):
            r += len(self.category[i])
            self.dis.append(dis[:, l: r])
            l += len(self.category[i])




