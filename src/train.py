import sys
from model import train_model
from model import adadelta,adagrad,sgd


#python train.py -d --dynet-weight-decay 0.

if __name__ == "__main__":
      assert (sys.argv[1]=="-d" or sys.argv[1]=="-t"), "-d dynet / -t theano"
      if sys.argv[1]=='-t': 
            print 'using theano'
            train_model(
                  max_epochs = 50,
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
      else:
            print 'using dynet'
            from dy_model import dy_train_model
            dy_train_model(
                  max_epochs = 50,
                  batch_size = 256,
                  ndims = 50,
                  nhiddens = 50,
                  dropout_rate = 0.2,
                  regularization = 0.000001,
                  margin_loss_discount = 0.2,
                  max_word_len = 4,
                  load_params = None,
                  start_point = 1,
                  max_sent_len = 60,
                  beam_size = 4,
                  shuffle_data = True,
                  train_file = '../data/pku_train_all',
                  dev_file = '../data/pku_test',
                  lr = 0.2,
                  pre_training = '../w2v/char_vecs_50',
                  early_update = True,
                  use_word_embed = True
                  )

