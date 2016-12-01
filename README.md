#CWS

This code implements the word segmentation algorithm proposed in the following paper.

Deng Cai and Hai Zhao, Neural Word Segmentation Learing for Chinese. ACL 2016.

##Update! a faster implementation using dynet as backend is now available.
##```python train.py -d``` to use the new (dynet based) version.

##Usage (old, also helpful to new version):
###- train
```python train.py -t```. To train a model, first check the hyperparameter settings in *train.py*. The training procedure will result a *config* file at the very beginning in which your hyperparameter settings are preserved, and output the trained model parameters to **\.npz* per epoch. 

###- test 
```python test.py params.npz input_file output_path config_file```. To test a trained model, specify the file that stores the model parameters as *params.npz* as well as the corresponding configuration file *config_file*. The test procedure will read data from *input_file* and output result to *output_path*.

###- evaluate     
For example, To see the best result (F1-score 95.5) on PKU dataset reported in our paper, first generate the output file through the trained model ( ```python test.py best_pku.npz ../data/pku_test somepath best_pku_config```), then use the command ```./score ../data/dic ../data/pku_test somepath```.
       
##Dependencies: 
Thanks to those excellent computing tools: Dynet, Theano, Numpy, Gensim.

##Author: 
[Deng Cai](https://jcyk.github.io/). Any question, feel free to contact me through [my email](mailto:thisisjcykcd@gmail.com).

##Citation:
If you find this code useful, please cite our paper.
```
@InProceedings{cai-zhao:2016:P16-1,
  author    = {Cai, Deng  and  Zhao, Hai},
  title     = {Neural Word Segmentation Learning for Chinese},
  booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  month     = {August},
  year      = {2016},
  address   = {Berlin, Germany},
  publisher = {Association for Computational Linguistics},
  pages     = {409--420},
  url       = {http://www.aclweb.org/anthology/P16-1039}
}
```

