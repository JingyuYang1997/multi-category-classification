import numpy as np
import pandas as pd
import re,os
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from config import Config
config_obj=Config()

'''该类对样本中的word信息进行关键词提取'''
class WordKeyVec():

    '''在初始化函数中分别读取三个文档中的word信息，赋予三个变量'''
    def __init__(self):
        self.train_alltext = []
        self.valid_alltext = []
        self.test_alltext = []
        self.pattern = '.*?\t.*?\t(.*?)\t.*?\t(.*?)\t.*?\t.*?\t.*?\n'
        self.pattern_test='.*?\t.*?\t(.*?)\t.*?\t(.*?)\n'
        with open('./data/train_b.txt') as f:
            train_data=f.read()
            train_word_text=re.findall(self.pattern,train_data)[1:]
            train_title_text=[item[0] for item in train_word_text]
            train_description_text=[item[1] for item in train_word_text]
            for index in range(len(train_title_text)):
                text=train_title_text[index]+','+train_description_text[index]
                text=text.split(',')
                text=" ".join(text)
                self.train_alltext.append(text)
        with open('./data/valid_b.txt') as f:
            valid_data=f.read()
            valid_word_text = re.findall(self.pattern, valid_data)[1:]
            valid_title_text = [item[0] for item in valid_word_text]
            valid_description_text = [item[1] for item in valid_word_text]
            for index in range(len(valid_title_text)):
                text=valid_title_text[index]+','+valid_description_text[index]
                text=text.split(',')
                text=" ".join(text)
                self.valid_alltext.append(text)
        with open('./data/test_b.txt') as f:
            test_data=f.read()
            test_word_text = re.findall(self.pattern_test, test_data)[1:]
            test_title_text = [item[0] for item in test_word_text]
            test_description_text = [item[1] for item in test_word_text]
            for index in range(len(test_title_text)):
                text=test_title_text[index]+','+test_description_text[index]
                text=text.split(',')
                text=" ".join(text)
                self.test_alltext.append(text)

    '''将三个文档中的所有样本结合，使用TF-IDF算法将各个样本中的word文本根据tf-idf值进行排序'''
    def save_key_words(self):
        word_alltext=self.train_alltext+self.valid_alltext+self.test_alltext
        vectorizer=CountVectorizer()
        TFmatrix=vectorizer.fit_transform(word_alltext)
        transformer=TfidfTransformer()
        weight=transformer.fit_transform(TFmatrix)
        weight=weight.astype(np.float32)                        # 得到词频矩阵weight
        weight=weight.toarray()
        words=vectorizer.get_feature_names()                    # 得到关键词词典words

        '''获得每个text中的topK个关键词'''
        keywords_list = []
        for i in range(len(word_alltext)):
            if i % 10000 == 0:
                print('已完成第%d个文档的关键词提取' % i)
            weighti = list(weight[i])
            tfidf_word = pd.DataFrame(words, columns=['word'])
            tfidf_weight = pd.DataFrame(weighti, columns=['weight'])
            word_weight = pd.concat([tfidf_word, tfidf_weight], axis=1)          # 将词汇列表与权重列表连接起来
            word_weight = word_weight.sort_values(by='weight', ascending=False)  # 按照权重对词汇列表进行降序排列
            keyword = np.array(word_weight['word'])
            keyword = keyword[0:config_obj.KEY_LEN]
            keyword = ",".join(keyword)
            keywords_list.append(keyword)

        '''分别得到三个文档每个样本中的关键词排序序列，并保存起来，以便后续使用（该项工作比较消耗时间和内存）'''
        train_keywords=keywords_list[0:len(self.train_alltext)]
        valid_keywords=keywords_list[len(self.train_alltext):len(self.train_alltext+self.valid_alltext)]
        test_keywords = keywords_list[len(self.train_alltext+self.valid_alltext):]
        with open('keys_extract/train_keywords.txt','w') as f:
            for item in train_keywords:
                f.write(item+'\n')
        with open('keys_extract/valid_keywords.txt','w') as f:
            for item in valid_keywords:
                f.write(item+'\n')
        with open('keys_extract/test_keywords.txt','w') as f:
            for item in test_keywords:
                f.write(item+'\n')


'''与word信息的关键词提取类似，对char信息进行关键字提取'''
'''剩余注释与上同理'''
class CharKeyVec():
    def __init__(self):
        self.train_alltext = []
        self.valid_alltext = []
        self.test_alltext = []
        self.pattern = '.*?\t.*?\t(.*?)\t.*?\t(.*?)\t.*?\t.*?\t.*?\n'
        self.pattern_test = '.*?\t.*?\t(.*?)\t.*?\t(.*?)\n'
        with open('./data/train_b.txt') as f:
            train_data = f.read()
            train_char_text = re.findall(self.pattern, train_data)[1:]
            train_title_text = [item[0] for item in train_char_text]
            train_description_text = [item[1] for item in train_char_text]
            for index in range(len(train_title_text)):
                text = train_title_text[index] + ',' + train_description_text[index]
                text = text.split(',')
                text = " ".join(text)
                self.train_alltext.append(text)
        with open('./data/valid_b.txt') as f:
            valid_data = f.read()
            valid_char_text = re.findall(self.pattern, valid_data)[1:]
            valid_title_text = [item[0] for item in valid_char_text]
            valid_description_text = [item[1] for item in valid_char_text]
            for index in range(len(valid_title_text)):
                text = valid_title_text[index] + ',' + valid_description_text[index]
                text = text.split(',')
                text = " ".join(text)
                self.valid_alltext.append(text)
        with open('./data/test_b.txt') as f:
            test_data = f.read()
            test_char_text = re.findall(self.pattern_test, test_data)[1:]
            test_title_text = [item[0] for item in test_char_text]
            test_description_text = [item[1] for item in test_char_text]
            for index in range(len(test_title_text)):
                text = test_title_text[index] + ',' + test_description_text[index]
                text = text.split(',')
                text = " ".join(text)
                self.test_alltext.append(text)

    def save_key_chars(self):
        char_alltext = self.train_alltext + self.valid_alltext + self.test_alltext
        vectorizer = CountVectorizer()
        TFmatrix = vectorizer.fit_transform(char_alltext)
        transformer = TfidfTransformer()
        weight = transformer.fit_transform(TFmatrix)
        weight = weight.astype(np.float32)
        weight = weight.toarray()
        chars = vectorizer.get_feature_names()  # 得到关键词词典
        # 获得每个text中的topK个关键词
        keychars_list = []
        for i in range(len(char_alltext)):
            if i % 10000 == 0:
                print('已完成第%d个文档的关键字提取' % i)
            weighti = list(weight[i])
            tfidf_char = pd.DataFrame(chars, columns=['char'])
            tfidf_weight = pd.DataFrame(weighti, columns=['weight'])
            char_weight = pd.concat([tfidf_char, tfidf_weight], axis=1)          # 将词汇列表与权重列表连接起来
            char_weight = char_weight.sort_values(by='weight', ascending=False)  # 按照权重进行降序排列
            keychar = np.array(char_weight['char'])
            keychar = keychar[0:config_obj.KEY_LEN]
            keychar = ",".join(keychar)
            keychars_list.append(keychar)
        train_keychars = keychars_list[0:len(self.train_alltext)]
        valid_keychars = keychars_list[len(self.train_alltext):len(self.train_alltext + self.valid_alltext)]
        test_keychars = keychars_list[len(self.train_alltext + self.valid_alltext):]
        with open('keys_extract/train_keychars.txt','w') as f:
            for item in train_keychars:
                f.write(item + '\n')
        with open('keys_extract/valid_keychars.txt','w') as f:
            for item in valid_keychars:
                f.write(item + '\n')
        with open('keys_extract/test_keychars.txt','w') as f:
            for item in test_keychars:
                f.write(item + '\n')


'''分配GPU '''
print('Assigning GPU……')
if config_obj.USE_GPA:
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    gpu_config.allow_soft_placement = True
    gpu_config.gpu_options.per_process_gpu_memory_fraction=0.6
    session = tf.Session(config=gpu_config)
print('GPU Assigned Successfully')

wordkey=WordKeyVec()
wordkey.save_key_words()
charkey=CharKeyVec()
charkey.save_key_chars()