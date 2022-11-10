import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import  DataLoader
import os
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter

class Embedding(nn.Module):
    def __init__(self, classnum, vector_size):
        super(Embedding, self).__init__()
        self.classnum = classnum
        self.fc1 = nn.Sequential(
            nn.Linear(self.classnum, 64),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(32, vector_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)
        return out


class FcMixModel(nn.Module):
    def __init__(self,cont_size, classnum, category, vector_size):
        super(FcMixModel, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.category = category
        self.embeding_block = []
        for i in range(len(self.category)):
            self.embeding_block.append(Embedding(len(self.category[i]),vector_size).to(self.device))

        self.fc1 = nn.Sequential(
            nn.Linear(cont_size+len(self.category)*vector_size , 128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128,64),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64,32),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(32, 16),
            torch.nn.Dropout(0.5),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU()
        )
        self.fc5 = nn.Linear(16, classnum)
        #
        # # # dim of Softmax should be 1!!!!! 0 will cause some error
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x_vector = [x[0]]
        for i in range(len(self.category)):
            x_vector.append(self.embeding_block[i](x[i+1]))
        x = torch.concat(x_vector,1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        out = self.softmax(x)
        return out



class FCMix:
    def __init__(self, cont_size= 8, classnum = 6, batchsize = 200, category = None, vector_size = 16, tesorboard_dir = './runs'+str(time.time())):
        self.cont_size = cont_size
        self.classnum = classnum
        self.batchsize = batchsize
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = FcMixModel(cont_size=self.cont_size, classnum=self.classnum, category = category,
                                vector_size=vector_size) .to(self.device)
        self.writer = SummaryWriter(log_dir=tesorboard_dir)

    def fit(self, train_db,valid_db, num_epochs, savepath, lr = 1e-4, gamma=0.99):
        train_loader = DataLoader(dataset=train_db, batch_size=self.batchsize, shuffle=False)
        valid_loader = DataLoader(dataset=valid_db, batch_size=self.batchsize, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[800, 1600, 5000, 8000], gamma=0.5)

        logdir = os.path.join(savepath, 'model_log.txt')
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        print('Start training')
        best_score = 0
        for epoch in range(num_epochs):
            for step, (inputs, labels) in enumerate(train_loader):
                (cont,[dis_1,dis_2,dis_3,dis_4,dis_5,dis_6,dis_7]) = inputs
                inputs = [cont.to(self.device),dis_1.to(self.device),dis_2.to(self.device),dis_3.to(self.device),
                          dis_4.to(self.device),dis_5.to(self.device),dis_6.to(self.device),dis_7.to(self.device)]
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels.squeeze(dim=1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                microf1_valid, macrof1_valid, loss_valid = self.check_accurancy(valid_loader, criterion)
                microf1_train, macrof1_train, loss_train = self.check_accurancy(train_loader, criterion)
                for param_group in optimizer.param_groups:
                    curr_lr = param_group['lr']
                self.writer.add_scalar('Loss/train', loss_train, epoch)
                self.writer.add_scalar('Loss/valid', loss_valid, epoch)
                self.writer.add_scalar('Microf1/train', microf1_train, epoch)
                self.writer.add_scalar('Microf1/valid', microf1_valid, epoch)
                self.writer.add_scalar('Macrof1/train', macrof1_train, epoch)
                self.writer.add_scalar('Macrof1/valid', macrof1_valid, epoch)
                self.writer.add_scalar('LR', curr_lr, epoch)
                print('Epoch [{}/{}], Loss: {:.4f} macrof1_train: {:.4f} microf1_train: {:.4f} '
                                'macrof1_valid: {:.4f} microf1_valid: {:.4f}\n'
                                .format(epoch + 1, num_epochs, loss.item(), macrof1_train, microf1_train,
                                        macrof1_valid, microf1_valid))
                if microf1_valid > best_score:
                    best_score = microf1_valid
                    PATH = os.path.join(savepath,'bestmodel.pth')
                    torch.save(self.model.state_dict(), PATH)
                if (epoch + 1) % 100 == 0:
                    with open(logdir, 'a+') as f:
                        f.write('Epoch [{}/{}], TrainingDataset: Loss: {:.4f} Macrof1: {:.4f} Microf1: {:.4f}\n'
                                'ValidDataset: Loss: {:.4f} Macrof1: {:.4f} Microf1: {:.4f}\n'
                                .format(epoch + 1, num_epochs, loss.item(), loss_train,macrof1_train, microf1_train,
                                        loss_valid, macrof1_valid, microf1_valid))
                    PATH = os.path.join(savepath, str(epoch + 1) + 'cnn' + '.pth')
                    torch.save(self.model.state_dict(), PATH)
        self.writer.close()

    def predict(self, data_loader, model_path):
        self.model.load_state_dict(torch.load(model_path))

        loader = DataLoader(dataset=data_loader, batch_size=self.batchsize, shuffle=False)
        with torch.no_grad():
            total_predcted = []
            total_labels = []
            for inputs, labels in loader:
                total_labels += labels.squeeze(dim=1)
                (cont, [dis_1, dis_2, dis_3, dis_4, dis_5, dis_6, dis_7]) = inputs
                inputs = [cont.to(self.device), dis_1.to(self.device), dis_2.to(self.device), dis_3.to(self.device),
                          dis_4.to(self.device), dis_5.to(self.device), dis_6.to(self.device), dis_7.to(self.device)]
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.to('cpu')
                total_predcted += predicted
        for i in range(len(total_predcted)):
            total_predcted[i] = total_predcted[i].item()
        try:
            macrof1 = f1_score(total_predcted, total_labels, average='macro')
            microf1 = f1_score(total_predcted, total_labels, average='micro')
            print('macrof1: {:.4f} microf1: {:.4f} \n'.format(macrof1, microf1))
        except:
            pass
        return total_predcted

    def check_accurancy(self, data_loader, criterion):
        with torch.no_grad():
            index = 0
            total_predcted = []
            total_labels = []
            total_loss = 0
            for inputs, labels in data_loader:
                total_labels += labels.squeeze(dim=1)
                index += 1
                (cont, [dis_1, dis_2, dis_3, dis_4, dis_5, dis_6, dis_7]) = inputs
                inputs = [cont.to(self.device), dis_1.to(self.device), dis_2.to(self.device), dis_3.to(self.device),
                          dis_4.to(self.device), dis_5.to(self.device), dis_6.to(self.device), dis_7.to(self.device)]
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels.squeeze(dim=1))
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.to('cpu')
                total_predcted += predicted
                total_loss += loss.item()
            macrof1 = f1_score(total_predcted, total_labels, average='macro')
            microf1 = f1_score(total_predcted, total_labels, average='micro')
            return microf1, macrof1, total_loss/index


