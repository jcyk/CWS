from model import train_model
from model import adadelta,adagrad,sgd
train_model(
            max_epoches = 50,
            optimizer = adagrad,
            batch_size = 256,
            ndims = 50,
            nhiddens = 50,
            dropout_rate = 0.2,
            regularization = 0.000001,
            margin_loss_discount = 0.2,
            max_word_len = 4,
            load_params = None,
            resume_training = False,
            start_point = 1,
            max_sent_len = 60,
            beam_size = 4,
            shuffle_data = True,
            train_file = '../data/pku_train',
            dev_file = '../data/pku_dev',
            lr = 0.2,
            pre_training = '../w2v/c_vecs_50'
            )
