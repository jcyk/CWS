# CWS

This is the implementation for the following paper.

Deng Cai and Hai Zhao, Neural Word Segmentation Learing for Chinese. Accepted by ACL 2016.

Author: Deng Cai

Any question, feel free to contact me through thisisjcykcd@gmail.com

Usage: 

       "python train.py" to train a model (See hyperparameter settings in train.py)

       "python test.py params.npz input_file output_path config_file" to test a trained model with (parameters in params.npz) and (corresponding configuration in config_file). 
       
       e.g., To see the best result on PKU dataset reported in our paper, first generate the output file through our trained model, i.e., type the command 'python test.py best_pku.npz ../data/pku_test somepath best_pku_config'. And then "./score ../data/dic ../data/pku_test somepath" as the output file will be saved in somepath already.
       
Dependencies: Numpy, Theano, Gensim
