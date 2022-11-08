import os
from model.nbc import NbcModel
from model.FC_model import FullConnectModel
from model.FC_mix import FCMix
from PrepareTool import PrepareTools, CommonDataset, MixDataset
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import f1_score

class DataProcess:
    def __init__(self, prepare_pattern, feature_type,csv_dir):
        self.pattern_dict = dict()
        self.pattern_dict['nbc'] = self.nbc
        self.pattern_dict['fc'] = self.dnn_common
        self.pattern_dict['fc_mix'] = self.dnn_mixdata

        self.prepare_pattern = prepare_pattern
        self.feature_type = feature_type
        self.train, self.valid, self.test = self.read_csv(csv_dir)

    def read_csv(self,csv_dir):
        train = pd.read_csv(os.path.join(csv_dir,'training.csv'))
        valid = pd.read_csv(os.path.join(csv_dir,'validation.csv'))
        test = pd.read_csv(os.path.join(csv_dir,'test.csv'))
        return train, valid, test

    def prepare_data(self):
        return self.pattern_dict[self.prepare_pattern]()


    def nbc(self):
        self.train = self.train.to_numpy()
        self.valid = self.valid.to_numpy()
        self.test = self.test.to_numpy()
        return [self.train, self.valid, self.test]

    def dnn_common(self):
        self.train, self.valid, self.test = PrepareTools.cls2onehot(
            self.train.to_numpy(),self.valid.to_numpy(),self.test.to_numpy(),feature_type,True)
        train_db, val_db, test_db = CommonDataset(self.train), CommonDataset(self.valid), CommonDataset(self.test)
        return [train_db, val_db, test_db]

    def dnn_mixdata(self):
        self.train, self.valid, self.test, dis_category = PrepareTools.cls2onehot(
            self.train.to_numpy(),self.valid.to_numpy(),self.test.to_numpy(),feature_type,False)
        train_db, val_db, test_db = MixDataset(self.train, dis_category), MixDataset(self.valid,dis_category), \
                                    MixDataset(self.test, dis_category)
        return [train_db, val_db, test_db, dis_category]

class Process:
    def __init__(self, prepare_pattern, feature_type,dataset):
        self.pattern_dict = dict()
        self.pattern_dict['nbc'] = self.nbc
        self.pattern_dict['fc'] = self.fc
        self.pattern_dict['fc_mix'] = self.fc_mix
        self.dataset = dataset
        self.train, self.valid, self.test = self.dataset[0], self.dataset[1], self.dataset[2]
        self.prepare_pattern = prepare_pattern
        self.pattern_dict[self.prepare_pattern]()

    def nbc(self):
        processor = NbcModel(feature_type, 6)
        processor.fit(self.train)
        pred_valid, label_valid = processor.predict(self.valid)
        print(f1_score(pred_valid, label_valid, average='macro'))
        print(f1_score(pred_valid, label_valid, average='micro'))
        pred_test, _= processor.predict(self.test)
        self.write_csv('Data2/validation.csv',pred_valid)
        self.write_csv('Data2/test.csv',pred_test)

    def fc(self):
        processor = FullConnectModel(input_size=29, classnum=6,
                                     tesorboard_dir = './result/Data2/fc_runs/1e-3')
        # processor.fit(self.train, self.valid, 10000, './result/Data2/fc',1e-3)
        # self.write_csv('Data2/validation.csv', processor.predict(self.valid, 'fc_model/10000cnn.pth'))
        # self.write_csv('Data2/test.csv', processor.predict(self.test, 'fc_model/10000cnn.pth'))

    def fc_mix(self):
        processor = FCMix(cont_size=8, classnum=6, category = self.dataset[-1], vector_size = 16,
                          tesorboard_dir = './result/Data2/fc_mix_runs/1e-4')
        processor.fit(self.train, self.valid, 10000, './result/Data2/fc_mix', 1e-4)
        # self.write_csv('Data2/validation.csv', processor.predict(self.valid, 'fc_mix/5000cnn.pth'))
        # self.write_csv('Data2/test.csv', processor.predict(self.test, 'fc_mix/5000cnn.pth'))



    def write_csv(self,csv_dir,pred_res):
        data = pd.read_csv(csv_dir)
        data['pred_label'] = pred_res
        data.to_csv(csv_dir.replace('.csv','pred.csv'), mode='a', index=False)



if __name__== "__main__" :
    feature_type = ['dis', 'cont', 'cont', 'dis', 'dis', 'cont', 'dis', 'cont', 'cont', 'dis', 'dis', 'cont', 'cont', 'cont', 'dis']
    dataprocessor = DataProcess(prepare_pattern='fc_mix', feature_type=feature_type, csv_dir='Data2')
    Process(prepare_pattern='fc_mix', dataset=dataprocessor.prepare_data(), feature_type=feature_type)
