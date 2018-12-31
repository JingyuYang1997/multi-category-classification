from keras.models import Model,Sequential
from keras.layers import Input, Dense, Conv1D, MaxPool1D,UpSampling1D,Flatten,BatchNormalization
from keras.layers import Concatenate
from keras.layers.embeddings import Embedding
from keras.layers import Dropout,LSTM,Bidirectional,GRU
from config import Config

config_obj = Config()

'''
得到stacking算法的第一级模型，为三个异构但预测效果差不多的基本模型，
分别为基于TextCNN的伪孪生网络M1，双向GRU网络M2，CNN与RNN串联的网络M3
'''

def get_first_models(cate_floor):
    NUM_CLASS = 0
    if cate_floor == 'f1':
        NUM_CLASS = config_obj.NUM_CLASS_1
    if cate_floor == 'f2':
        NUM_CLASS = config_obj.NUM_CLASS_2
    if cate_floor == 'f3':
        NUM_CLASS = config_obj.NUM_CLASS_3

    '''Siamese_TextCNN model'''
    # Word encoding layer(embedding+conv1D）
    # Word Embedding
    word_input_shape=(config_obj.KEY_LEN,)
    word_input=Input(shape=word_input_shape)
    word_embedding=Embedding(input_dim=config_obj.VOCAB_SIZE,output_dim=config_obj.EMBEDDING_DIM, \
                             input_length=config_obj.KEY_LEN,name='word_embedding')(word_input)
    dropout1=Dropout(0.4)(word_embedding)

    # Word Convolutional layer
    word_conv_blockes = []
    ks = config_obj.KERNELSIZES
    wordconv1 = Conv1D(filters=config_obj.NUM_FILTERS, kernel_size=ks[0], padding='valid', activation='relu',\
                       strides=1)(dropout1)
    wordconv1 = BatchNormalization()(wordconv1)
    word_encoded1 = MaxPool1D(pool_size=2)(wordconv1)
    wordconv1 = Dropout(0.5)(word_encoded1)
    wordconv1 = Flatten()(wordconv1)
    word_conv_blockes.append(wordconv1)

    wordconv2 = Conv1D(filters=config_obj.NUM_FILTERS, kernel_size=ks[1], padding='valid', \
                       activation='relu', strides=1)(dropout1)
    wordconv2 = BatchNormalization()(wordconv2)
    word_encoded2 = MaxPool1D(pool_size=2)(wordconv2)
    wordconv2 = Dropout(0.5)(word_encoded2)
    wordconv2 = Flatten()(wordconv2)
    word_conv_blockes.append(wordconv2)

    wordconv3 = Conv1D(filters=config_obj.NUM_FILTERS, kernel_size=ks[2], padding='valid', \
                       activation='relu', strides=1)(dropout1)
    wordconv3 = BatchNormalization()(wordconv3)
    word_encoded3 = MaxPool1D(pool_size=2, name='word_encoded3')(wordconv3)
    wordconv3 = Dropout(0.5)(word_encoded3)
    wordconv3 = Flatten()(wordconv3)
    word_conv_blockes.append(wordconv3)

    word_encoded = Concatenate(name='words_concate')(word_conv_blockes) if len(word_conv_blockes) > 1 else word_conv_blockes[0]
    word_encoded = Dropout(0.2)(word_encoded)
    # dropout2=Dropout(0.4)(conv)

    # Character encoding layer(embedding+conv1D)
    # Character Embedding
    char_input_shape = (config_obj.KEY_LEN,)
    char_input = Input(shape=char_input_shape)
    char_embedding = Embedding(input_dim=config_obj.CHARVOC_SIZE, output_dim=config_obj.EMBEDDING_DIM, \
                               input_length=config_obj.KEY_LEN)(char_input)
    dropout1 = Dropout(0.4)(char_embedding)

    # Character Convolutional layer
    char_conv_blockes = []
    for sz in config_obj.KERNELSIZES:
        conv = Conv1D(filters=config_obj.NUM_FILTERS, kernel_size=sz, padding='valid', \
                      activation='relu', strides=1)(dropout1)
        conv = BatchNormalization()(conv)
        conv = MaxPool1D(pool_size=2)(conv)
        conv = Dropout(0.5)(conv)
        conv = Flatten()(conv)
        char_conv_blockes.append(conv)
    char_encoded = Concatenate(name='char_concate')(char_conv_blockes) if len(char_conv_blockes) > 1 else char_conv_blockes[0]
    # dropout2=Dropout(0.4)(conv)

    # to build the Siamese Network
    concate=Concatenate(axis=-1,name='Siamese_concate')([word_encoded, char_encoded])

    # '''decoder'''
    # word_decoder = Conv1D(filters=config_obj.NUM_FILTERS, kernel_size=4, padding='valid',activation='relu', \
    #                     strides=1,name='decoder_conv1d')(word_encoded3)
    # word_decoder = BatchNormalization(name='decoder_batchnorm')(word_decoder)
    # word_decoder = UpSampling1D(size=3)(word_decoder)
    # word_decoder = Conv1D(filters=1,kernel_size=5,padding='valid',activation='sigmoid',strides=1)(word_decoder)
    # word_decoder = Flatten(name='reconstruct')(word_decoder)
    # model_textcnn= Model([word_input,char_input],[model_output,word_decoder])
    # losses = {'classification':'categorical_crossentropy','reconstruct':'mse'}
    # lossweights={'classification':1.0,'reconstruct':5.0}
    # model_textcnn.compile(optimizer='adam', loss=losses, loss_weights=lossweights,metrics=None)

    # FC layer
    dense = Dense(config_obj.HIDDEN_DIMS, activation='relu')(concate)
    dropout3=Dropout(0.4)(dense)
    model_output = Dense(NUM_CLASS,activation='softmax',name='classification')(dropout3)
    model_textcnn=Model([word_input,char_input],model_output)
    model_textcnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=None)

    '''Bi-GRU model'''
    gru_input_shape=(config_obj.KEY_LEN,)
    gru_input=Input(shape=gru_input_shape)
    gru=Embedding(input_dim=config_obj.VOCAB_SIZE,output_dim=config_obj.EMBEDDING_DIM, \
                             input_length=config_obj.KEY_LEN)(gru_input)
    gru=Bidirectional(GRU(256,dropout=0.5,recurrent_dropout=0.5,return_sequences=True))(gru)
    gru=Bidirectional(GRU(256,dropout=0.5,recurrent_dropout=0.1))(gru)
    gru=Dense(config_obj.HIDDEN_DIMS,activation='relu')(gru)
    gru=Dropout(0.5)(gru)
    gru_output =Dense(NUM_CLASS,activation='softmax')(gru)
    model_gru=Model(gru_input,gru_output)
    model_gru.compile(optimizer='adam',loss='categorical_crossentropy',metrics=None)

    '''CNN+RNN model'''
    cr_input_shape = (config_obj.KEY_LEN,)
    cr_input = Input(shape=cr_input_shape)
    cr=Embedding(input_dim=config_obj.VOCAB_SIZE,output_dim=config_obj.EMBEDDING_DIM,input_length=config_obj.KEY_LEN)(cr_input)
    cr=Conv1D(256, 3, padding='same',activation='relu',strides=1)(cr)
    cr=MaxPool1D(pool_size=2)(cr)
    cr=Dropout(0.5)(cr)
    cr=GRU(256, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)(cr)
    cr=GRU(256, dropout=0.5, recurrent_dropout=0.5)(cr)
    cr_output=Dense(NUM_CLASS, activation='softmax')(cr)
    model_cr=Model(cr_input,cr_output)
    model_cr.compile(optimizer='adam',loss='categorical_crossentropy',metrics=None)

    models={'textcnn':model_textcnn,'bi_gru':model_gru,'cnn+rnn':model_cr}
    return models


'''
stacking算法的第二级模型，深层卷积网络M4
'''
def get_second_model(cate_floor):
    if cate_floor == 'f1':
        NUM_CLASS = config_obj.NUM_CLASS_1
    if cate_floor == 'f2':
        NUM_CLASS = config_obj.NUM_CLASS_2
    if cate_floor == 'f3':
        NUM_CLASS = config_obj.NUM_CLASS_3
    input_shape=(3*NUM_CLASS,1,)
    model_input=Input(shape=input_shape)
    conv1=Conv1D(filters=128,kernel_size=3,padding='valid',activation='relu',strides=1)(model_input)
    conv1=BatchNormalization()(conv1)
    conv1=MaxPool1D(pool_size=2)(conv1)
    dropout1=Dropout(0.2)(conv1)
    conv2=Conv1D(filters=64,kernel_size=3,padding='valid',activation='relu',strides=1)(dropout1)
    conv2=BatchNormalization()(conv2)
    conv2=MaxPool1D(pool_size=2)(conv2)
    dropout2=Dropout(0.2)(conv2)
    flatten=Flatten()(dropout2)
    dense1=Dense(units=64,activation='relu')(flatten)
    dropout3=Dropout(0.2)(dense1)
    dense2=Dense(units=32,activation='relu')(dropout3)
    model_output=Dense(units=NUM_CLASS,activation='softmax')(dense2)
    model=Model(model_input,model_output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=None)
    return model