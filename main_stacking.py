import numpy as np
import math
import os
import tensorflow as tf
from keras.utils import to_categorical
from config import Config
from load_sorted_data import get_word_data,get_char_data
from load_models import get_first_models,get_second_model
from evaluation import Accurancy
from keras.callbacks import LearningRateScheduler
from keras import backend as Kb
config_obj = Config()


'''分配GPU'''
print('Assigning GPU……')
if config_obj.USE_GPA:
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    gpu_config.allow_soft_placement = True
    # gpu_config.gpu_options.per_process_gpu_memory_fraction=0.6
    session = tf.Session(config=gpu_config)
print('GPU Assigned Successfully')


'''载入编码后的数据'''
print('Loading Data Please Wait……')
train_word_data,train_labels=get_word_data('train')
val_word_data,val_labels=get_word_data('valid')
train_char_data=get_char_data('train')
val_char_data=get_char_data('valid')
train_label1=to_categorical(train_labels[0],num_classes=config_obj.NUM_CLASS_1)
train_label2=to_categorical(train_labels[1],num_classes=config_obj.NUM_CLASS_2)
train_label3=to_categorical(train_labels[2],num_classes=config_obj.NUM_CLASS_3)
val_label1=to_categorical(val_labels[0],num_classes=config_obj.NUM_CLASS_1)
val_label2=to_categorical(val_labels[1],num_classes=config_obj.NUM_CLASS_2)
val_label3=to_categorical(val_labels[2],num_classes=config_obj.NUM_CLASS_3)
print('Loading Data Successfully!')

ac_1 = Accurancy(config_obj.NUM_CLASS_1)
ac_2 = Accurancy(config_obj.NUM_CLASS_2)
ac_3 = Accurancy(config_obj.NUM_CLASS_3)

'''将训练集和验证集打乱，方便之后的K折交叉验证分配数据'''
permutation=np.random.permutation(len(train_word_data))
train_word_data=train_word_data[permutation]
train_char_data=train_char_data[permutation]
train_label1=train_label1[permutation]
train_label2=train_label2[permutation]
train_label3=train_label3[permutation]


'''随着训练不断进行使学习率下降'''
def step_decay(epoch_index):
    initial_lrate = config_obj.LR
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch_index)/epochs_drop))  # floor取下舍整数
    return lrate


learning_rate = LearningRateScheduler(step_decay)

'''跟概率形式的预测结果得到标签，并转化为one-hot编码，方便计算分数'''
def prob2code(output_prob):
    outputs_list = []
    for item in output_prob:
        output = np.zeros(item.shape[0])
        output[np.argmax(item)] = 1
        outputs_list.append(output)
    outputs_list = np.array(outputs_list)
    outputs_list.astype(np.float32)
    return outputs_list


'''设定交叉验证为5个cross'''
K=5
data_len=math.floor(len(train_word_data)/K)
train_word_data_list=list(train_word_data)
train_char_data_list=list(train_char_data)


'''stacking算法训练第一层的基本分类器'''
def stacking_first_train(cate_floor):
    if cate_floor == 'f1':
        train_label = train_label1
    if cate_floor == 'f2':
        train_label=train_label2
    if cate_floor == 'f3':
        train_label = train_label3
    train_label_list = list(train_label)
    M1_train_pred=[]
    M2_train_pred=[]
    M3_train_pred=[]
    M1_valid_pred=[]
    M2_valid_pred=[]
    M3_valid_pred=[]
    print('first-class models related to ' + cate_floor + ' are being trained……')

    '''对训练集进行5折交叉验证训练，每折训练中进行验证集的预测并在最后取算术平均'''
    for k in range(K):
        '''载入第一层模型'''
        model_dict=get_first_models(cate_floor)
        M1 = model_dict['textcnn']
        M2 = model_dict['bi_gru']
        M3 = model_dict['cnn+rnn']
        if k==K-1:
            train_pred_word = np.array(train_word_data_list[k*data_len:])
            train_pred_char = np.array(train_char_data_list[k*data_len:])
            train_learn_word = np.array(train_word_data_list[0:k*data_len])
            train_learn_char = np.array(train_char_data_list[0:k*data_len])
            train_learn_label = np.array(train_label_list[0:k*data_len])
        if k<K-1:
            train_pred_word=np.array(train_word_data_list[k*data_len:(k+1)*data_len])
            train_pred_char=np.array(train_char_data_list[k*data_len:(k+1)*data_len])
            train_learn_word=np.array(train_word_data_list[0:k*data_len]+train_word_data_list[(k+1)*data_len:])
            train_learn_char=np.array(train_char_data_list[0:k*data_len]+train_char_data_list[(k+1)*data_len:])
            train_learn_label=np.array(train_label_list[0:k*data_len]+train_label_list[(k+1)*data_len:])
        print('M1_' + cate_floor + ' is being trained in the ' + str(k + 1) + ' cross')
        M1.fit([train_learn_word, train_learn_char], train_learn_label, batch_size=config_obj.TRAIN_BATCH_SIZE, epochs= \
               config_obj.NUM_EPOCHS, verbose=2, callbacks=[learning_rate])
        # ''' Autoencoder'''
        # M1.fit([train_learn_word, train_learn_char], [train_learn_label,train_learn_word], batch_size=config_obj.\
        #        TRAIN_BATCH_SIZE, epochs=config_obj.NUM_EPOCHS, verbose=2, callbacks=[learning_rate])
        print('M2_' + cate_floor + ' is being trained in the ' + str(k + 1) + ' cross')
        M2.fit(train_learn_word, train_learn_label, batch_size=config_obj.TRAIN_BATCH_SIZE, epochs=config_obj.NUM_EPOCHS,\
               verbose=2, callbacks=[learning_rate])
        print('M3_' + cate_floor + ' is being trained in the ' + str(k + 1) + ' cross')
        M3.fit(train_learn_word, train_learn_label, batch_size=config_obj.TRAIN_BATCH_SIZE, epochs=config_obj.NUM_EPOCHS,\
               verbose=2, callbacks=[learning_rate])
        M1_train_pred.append(M1.predict([train_pred_word, train_pred_char]))
        # '''Autoencoder'''
        # M1_train_pred.append(M1.predict([train_pred_word, train_pred_char])[0])
        M2_train_pred.append(M2.predict(train_pred_word))
        M3_train_pred.append(M3.predict(train_pred_word))
        M1_valid_pred.append(M1.predict([val_word_data, val_char_data]))
        M2_valid_pred.append(M2.predict(val_word_data))
        M3_valid_pred.append(M3.predict(val_word_data))
        if k == K-1:
            M1.save('./models/model_'+cate_floor+'_M1.h5')
            M2.save('./models/model_'+cate_floor+'_M2.h5')
            M3.save('./models/model_'+cate_floor+'_M3.h5')
            print('first-class models related to'+cate_floor+'have been trained and saved!')
        Kb.clear_session()

    '''P,T分别为stacking算法第二层模型的训练集和验证集'''
    P1 = np.vstack((M1_train_pred[0], M1_train_pred[1], M1_train_pred[2], M1_train_pred[3],M1_train_pred[4]))
    P2 = np.vstack((M2_train_pred[0], M2_train_pred[1], M2_train_pred[2], M2_train_pred[3],M2_train_pred[4]))
    P3 = np.vstack((M3_train_pred[0], M3_train_pred[1], M3_train_pred[2], M3_train_pred[3],M3_train_pred[4]))
    P = np.hstack((P1,P2,P3))
    T1 = (M1_valid_pred[0] + M1_valid_pred[1] + M1_valid_pred[2] + M1_valid_pred[3]+M1_valid_pred[4])/5
    T2 = (M2_valid_pred[0] + M2_valid_pred[1] + M2_valid_pred[2] + M2_valid_pred[3]+M2_valid_pred[4])/5
    T3 = (M3_valid_pred[0] + M3_valid_pred[1] + M3_valid_pred[2] + M3_valid_pred[3]+M3_valid_pred[4])/5
    T = np.hstack((T1,T2,T3))
    P = np.expand_dims(P,axis=-1)
    T = np.expand_dims(T,axis=-1)
    return P, T


'''stacking算法训练第二层的强分类器'''
def stacking_second_train(cate_floor,P,T):
    if cate_floor == 'f1':
        train_label = train_label1
        val_label = val_label1
        ac=ac_1
    if cate_floor == 'f2':
        train_label=train_label2
        val_label = val_label2
        ac=ac_2
    if cate_floor == 'f3':
        train_label = train_label3
        val_label = val_label3
        ac=ac_3
    M4 = get_second_model(cate_floor)
    print('M4_' + cate_floor + ' is being trained ')
    M4.fit(P,train_label,batch_size=config_obj.TRAIN_BATCH_SIZE,epochs=config_obj.NUM_EPOCHS,verbose=2, \
           validation_data=[T,val_label],callbacks=[learning_rate])
    M4.save('./models/model_'+cate_floor+'_M4.h5')
    print('second-class model related to ' + cate_floor + ' have been trained and saved!')
    print('calculating the ' + cate_floor + ' score on train_data')
    outputs_prob_train=M4.predict(P)
    outputs_train=prob2code(outputs_prob_train)
    ac.save_data(outputs_train,train_label)
    f1_train=ac.caculate_f1()
    print('calculating the ' + cate_floor + ' score on valid_data')
    outputs_prob_valid = M4.predict(T)
    outputs_valid = prob2code(outputs_prob_valid)
    ac.save_data(outputs_valid, val_label)
    f1_valid = ac.caculate_f1()
    print('all tasks on '+cate_floor+' has been completed!')
    return f1_train,f1_valid


P1,T1=stacking_first_train('f1')
P2,T2=stacking_first_train('f2')
P3,T3=stacking_first_train('f3')
F1_cate1_train,F1_cate1_valid=stacking_second_train('f1',P1,T1)
F1_cate2_train,F1_cate2_valid=stacking_second_train('f2',P2,T2)
F1_cate3_train,F1_cate3_valid=stacking_second_train('f3',P3,T3)

F1_train= 0.1*F1_cate1_train+0.3*F1_cate2_train+0.6*F1_cate3_train
F1_valid= 0.1*F1_cate1_valid+0.3*F1_cate2_valid+0.6*F1_cate3_valid

print('epoch '+str(config_obj.NUM_EPOCHS)+' training has finished')
print(60 * '*')
print('     Training Dataset Evaluation:')
print('             F1_cate1=' + str(F1_cate1_train))
print('             F1_cate2=' + str(F1_cate2_train))
print('             F1_cate3=' + str(F1_cate3_train))
print('             TrainScore=' + str(F1_train))
print(60 * '*')
print('     Validation Dataset Evaluation:')
print('             F1_cate1=' + str(F1_cate1_valid))
print('             F1_cate2=' + str(F1_cate2_valid))
print('             F1_cate3=' + str(F1_cate3_valid))
print('             ValidScore=' + str(str(F1_valid)))

print('Models has been trained,validated and saved successfully!')