class Config(object):
    def __init__(self):
        self.USE_GPA = True
        self.NUM_EPOCHS = 1
        self.TRAIN_BATCH_SIZE = 256
        self.VAL_BATCH_SIZE = 256
        self.TEST_BATCH_SIZE = 256

        self.LR = 1e-3

        self.NUM_CLASS_1 = 10
        self.NUM_CLASS_2 = 64
        self.NUM_CLASS_3 = 125
        self.EMBEDDING_DIM = 100
        self.VOCAB_SIZE = 209209
        self.CHARVOC_SIZE = 6118
        self.KERNELSIZES = (3,5,8)
        self.NUM_FILTERS = 128
        self.HIDDEN_DIMS = 1024
        self.KEY_LEN = 100
