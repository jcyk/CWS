# -*- coding: UTF-8 -*-
import numpy as np
import theano
import re
import sys
import json

import tools
def output_result(Y,table,path,filename):
    def seg(lens,characters):
        res = []
        begin = 0
        for len in lens:
            res.append(''.join(characters[begin:begin+len]))
            begin+=len
        return res

    fi = open(path)
    fo = open(filename,'wb')
    Y_idx = 0
    for line in fi.readlines():
        sent = unicode(line.decode('utf8')).split()
        Left = 0
        output_sent = []
        for idx,word in enumerate(sent):
            if len(re.sub('\W','',word,flags=re.U))==0:
                if idx>Left:
                    words =seg(Y[table[Y_idx]],list(''.join(sent[Left:idx])))
                    output_sent.extend(words)
                    Y_idx+=1
                Left = idx+1
                output_sent.append(word)
        if Left!=len(sent):
            words = seg(Y[table[Y_idx]],list(''.join(sent[Left:])))
            output_sent.extend(words)
            Y_idx+=1
        fo.write('  '.join(output_sent).encode('utf8')+'\r\n')
    fi.close()
    fo.close()

def test(character_idx_map,
         options,
         params,
         path,
         filename,
         batch_size = 512
         ):
    
    X = tools.prepareData(character_idx_map,path,test=True)
    dropout = (1-options['dropout_rate'])*np.ones((options['ndims'],), dtype=theano.config.floatX)
    start,n = 0,len(X)
    idx_list = range(n)
    lens = [len(x) for x in X]
    idx_list = sorted(idx_list,cmp = lambda x,y: cmp(lens[x],lens[y]))
    Y = []
    print 'count_test_sentences',len(X)
    
    for i in range(n//batch_size):
        batch_idx = idx_list[start:start+batch_size]
        x = [X[t] for t in batch_idx]
        x_lens = [lens[t] for t in batch_idx]
        x = tools.asMatrix(x)
        sY = tools.segment(params,options,x,x_lens,dropout)
        Y.extend(sY)
        start+=batch_size
    if start!=n:
        batch_idx = idx_list[start:]
        x = [X[t] for t in batch_idx]
        x_lens = [lens[t] for t in batch_idx]
        x = tools.asMatrix(x)
        sY = tools.segment(params,options,x,x_lens,dropout)
        Y.extend(sY)
    table = {}
    nb= 0
    for idx in idx_list:
        table[idx] = nb
        nb+=1
    output_result(Y,table,path,filename)

def testFromFile(load_params,
                 path,
                 filename,
                 config = 'config'):
    '''
        path -> filename with load_params and config
    '''
    config = json.loads(open(config).read())
    character_idx_map = config['character_idx_map']
    options = config['options']
    params = np.load(load_params)
    test(character_idx_map,options,params,path,filename)


if __name__ == '__main__':
    '''
        python test.py params input result config
    '''
    if len(sys.argv)>=5:
        testFromFile(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
    else:
        testFromFile(sys.argv[1],sys.argv[2],sys.argv[3])
