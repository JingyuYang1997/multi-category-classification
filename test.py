from keras.models import load_model
import os
import re
import tensorflow as tf
import numpy as np
from word_index import WordIndex
from class_index import ClassIndex
from config import Config
from load_sorted_data import get_word_data,get_char_data

config_obj=Config()
classindex=ClassIndex(opt=config_obj)
wordindex=WordIndex(opt=config_obj)
charindex=ClassIndex(opt=config_obj)

print('Assigning GPU……')
if config_obj.USE_GPA:
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    gpu_config.allow_soft_placement = True
    # gpu_config.gpu_options.per_process_gpu_memory_fraction=0.6
    session = tf.Session(config=gpu_config)
print('GPU Assigned Successfully')

print('loading test data……')
test_word_data=get_word_data('test')
test_char_data=get_char_data('test')
pattern='(.*?)\t.*?\t.*?\t.*?\t.*?\n'
with open('./data/test_b.txt') as f:
    data=f.read()
    data=re.findall(pattern,data)[1:]
    test_ids=[item for item in data]
print('Loading Test Data Successfully!')

print('loading model……')
model_f1_M1=load_model('model_f1_M1.h5')
model_f1_M2=load_model('model_f1_M2.h5')
model_f1_M3=load_model('model_f1_M3.h5')
model_f1_M4=load_model('model_f1_M4.h5')

model_f2_M1=load_model('model_f2_M1.h5')
model_f2_M2=load_model('model_f2_M2.h5')
model_f2_M3=load_model('model_f2_M3.h5')
model_f2_M4=load_model('model_f2_M4.h5')

model_f3_M1=load_model('model_f3_M1.h5')
model_f3_M2=load_model('model_f3_M2.h5')
model_f3_M3=load_model('model_f3_M3.h5')
model_f3_M4=load_model('model_f3_M4.h5')
print('Loading Model Successfully!')

T1_f1=model_f1_M1.predict([test_word_data,test_char_data])
T2_f1=model_f1_M2.predict(test_word_data)
T3_f1=model_f1_M3.predict(test_word_data)
T1_f2=model_f2_M1.predict([test_word_data,test_char_data])
T2_f2=model_f2_M2.predict(test_word_data)
T3_f2=model_f2_M3.predict(test_word_data)
T1_f3=model_f3_M1.predict([test_word_data,test_char_data])
T2_f3=model_f3_M2.predict(test_word_data)
T3_f3=model_f3_M3.predict(test_word_data)

T_f1=np.hstack((T1_f1,T2_f1,T3_f1))
T_f1 = np.expand_dims(T_f1,axis=-1)
T_f2=np.hstack((T1_f2,T2_f2,T3_f2))
T_f2 = np.expand_dims(T_f2,axis=-1)
T_f3=np.hstack((T1_f3,T2_f3,T3_f3))
T_f3 = np.expand_dims(T_f3,axis=-1)

test_prob1=model_f1_M4.predict(T_f1)
test_prob2=model_f2_M4.predict(T_f2)
test_prob3=model_f3_M4.predict(T_f3)


def prob2label(prob):
    labels=[]
    for item in prob:
        label=np.argmax(item)
        labels.append(label)
    return np.array(labels)


test_label1=prob2label(test_prob1)
test_label2=prob2label(test_prob2)
test_label3=prob2label(test_prob3)

with open('./data/ans.txt','w') as f:
    f.write("item_id\tcate1_id\tcate2_id\tcate3_id\n")
    for index in range(test_label1.shape[0]):
        class1_index=test_label1[index]
        class2_index=test_label2[index]
        class3_index=test_label3[index]
        class_index=[class1_index,class2_index,class3_index]
        class_labels=classindex.idx_to_class(class_index)
        answer=test_ids[index]+'\t'+class_labels[0]+'\t' + \
               class_labels[1]+'\t'+class_labels[2]+'\n'
        f.write(answer)
