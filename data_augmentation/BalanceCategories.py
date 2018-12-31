from DataLoader import MyDataSet_BalanceCategories
from config import Config
import numpy as np

opt = Config()


data_tr_b = MyDataSet_BalanceCategories(opt.TRAIN_FILE_B)
data_va_b = MyDataSet_BalanceCategories(opt.VALID_FILE_B)

data_all = data_tr_b + data_va_b


class BalanceCategories3():
    def __init__(self, opt):
        self.opt = opt
        self.classifications= []
        self.length = []

    def classify_3(self, data):
        data.sort(key=lambda x: x[7])       # 用class_3排序
        class_3 = []
        classification = []
        for sentence in data:
            class_3.extend([sentence[7]])
            if len(class_3) == 1 or class_3[-1] == class_3[-2]:
                classification.append(sentence)
            else:
                self.classifications.append(classification)
                self.length.extend([len(classification)])
                class_3 = [sentence[7]]
                classification = [sentence]
        self.classifications.append(classification)
        self.length.extend([len(classification)])


    def write_balance_3(self, num_data=1.2e6):
        num_class_3 = int(num_data//self.opt.NUM_CLASS_3)
        with open('./data/train_balance_class3.txt', 'w') as f:
            f.write("item_id\ttitle_characters\ttitle_words\tdescription_characters\tdescription_words\tcate1_id\tcate2_id\tcate3_id\n")
            for i in range(self.opt.NUM_CLASS_3):
                if self.length[i]>=num_class_3:
                    for index in range(self.length[i]):
                        sen = self.classifications[i][index]
                        wr = sen[0] + '\t' + sen[1] + '\t' + sen[2] + '\t' + sen[3] + '\t' + sen[4] + '\t' + sen[
                            5] + '\t' + sen[6] + '\t' + sen[7] + '\n'
                        f.write(wr)
                else:
                    for j in range(num_class_3):
                        idx = j % self.length[i]
                        sen = self.classifications[i][idx]
                        title_word=sen[2].split(',')
                        title_permutation=np.random.permutation(len(title_word))
                        title_words=np.array(title_word)
                        title_words=title_words[title_permutation]
                        title_words=list(title_words)
                        sen2=",".join(title_words)
                        description_words=sen[4].split(',')
                        description_words=np.array(description_words)
                        description_permutation=np.random.permutation(len(description_words))
                        description_words=description_words[description_permutation]
                        description_words=list(description_words)
                        sen4=",".join(description_words)
                        wr = sen[0]+'\t'+sen[1]+'\t'+sen2+'\t'+sen[3]+'\t'+sen4+'\t'+sen[5]+'\t'+sen[6]+'\t'+sen[7]+'\n'
                        f.write(wr)


class BalanceCategories2():
    def __init__(self, opt):
        self.opt = opt
        self.classifications= []
        self.length = []

    def classify_2(self, data):
        data.sort(key=lambda x: x[6])       # 用class_3排序
        class_2 = []
        classification = []
        for sentence in data:
            class_2.extend([sentence[6]])
            if len(class_2) == 1 or class_2[-1] == class_2[-2]:
                classification.append(sentence)
            else:
                self.classifications.append(classification)
                self.length.extend([len(classification)])
                class_2 = [sentence[6]]
                classification = [sentence]
        self.classifications.append(classification)
        self.length.extend([len(classification)])


    def write_balance_2(self, num_data=2e6):
        num_class_2 = int(num_data//self.opt.NUM_CLASS_2)
        with open('./data/train_balance_class2.txt', 'w') as f:
            f.write("item_id\ttitle_characters\ttitle_words\tdescription_characters\tdescription_words\tcate1_id\tcate2_id\tcate3_id\n")
            for i in range(self.opt.NUM_CLASS_2):
                if self.length[i]>=num_class_2:
                    for index in range(self.length[i]):
                        sen = self.classifications[i][index]
                        wr = sen[0] + '\t' + sen[1] + '\t' + sen[2] + '\t' + sen[3] + '\t' + sen[4] + '\t' + sen[
                            5] + '\t' + sen[6] + '\t' + sen[7] + '\n'
                        f.write(wr)
                else:
                    for j in range(num_class_2):
                        idx = j % self.length[i]
                        sen = self.classifications[i][idx]
                        title_word=sen[2].split(',')
                        title_permutation=np.random.permutation(len(title_word))
                        title_words=np.array(title_word)
                        title_words=title_words[title_permutation]
                        title_words=list(title_words)
                        sen2=",".join(title_words)
                        description_words=sen[4].split(',')
                        description_words=np.array(description_words)
                        description_permutation=np.random.permutation(len(description_words))
                        description_words=description_words[description_permutation]
                        description_words=list(description_words)
                        sen4=",".join(description_words)
                        wr = sen[0]+'\t'+sen[1]+'\t'+sen2+'\t'+sen[3]+'\t'+sen4+'\t'+sen[5]+'\t'+sen[6]+'\t'+sen[7]+'\n'
                        f.write(wr)

class BalanceCategories1():
    def __init__(self, opt):
        self.opt = opt
        self.classifications= []
        self.length = []

    def classify_1(self, data):
        data.sort(key=lambda x: x[5])       # 用class_3排序
        class_1 = []
        classification = []
        for sentence in data:
            class_1.extend([sentence[5]])
            if len(class_1) == 1 or class_1[-1] == class_1[-2]:
                classification.append(sentence)
            else:
                self.classifications.append(classification)
                self.length.extend([len(classification)])
                class_1 = [sentence[5]]
                classification = [sentence]
        self.classifications.append(classification)
        self.length.extend([len(classification)])


    def write_balance_1(self, num_data=2e6):
        num_class_1 = int(num_data//self.opt.NUM_CLASS_1)
        with open('./data/train_balance_class1.txt', 'w') as f:
            f.write("item_id\ttitle_characters\ttitle_words\tdescription_characters\tdescription_words\tcate1_id\tcate2_id\tcate3_id\n")
            for i in range(self.opt.NUM_CLASS_1):
                if self.length[i]>=num_class_1:
                    for index in range(self.length[i]):
                        sen = self.classifications[i][index]
                        wr = sen[0] + '\t' + sen[1] + '\t' + sen[2] + '\t' + sen[3] + '\t' + sen[4] + '\t' + sen[
                            5] + '\t' + sen[6] + '\t' + sen[7] + '\n'
                        f.write(wr)
                else:
                    for j in range(num_class_1):
                        idx = j % self.length[i]
                        sen = self.classifications[i][idx]
                        title_word=sen[2].split(',')
                        title_permutation=np.random.permutation(len(title_word))
                        title_words=np.array(title_word)
                        title_words=title_words[title_permutation]
                        title_words=list(title_words)
                        sen2=",".join(title_words)
                        description_words=sen[4].split(',')
                        description_words=np.array(description_words)
                        description_permutation=np.random.permutation(len(description_words))
                        description_words=description_words[description_permutation]
                        description_words=list(description_words)
                        sen4=",".join(description_words)
                        wr = sen[0]+'\t'+sen[1]+'\t'+sen2+'\t'+sen[3]+'\t'+sen4+'\t'+sen[5]+'\t'+sen[6]+'\t'+sen[7]+'\n'
                        f.write(wr)
BC = BalanceCategories2(opt)
BC.classify_2(data_all)
BC.write_balance_2(num_data=2e6)        # num_data: 数据量
