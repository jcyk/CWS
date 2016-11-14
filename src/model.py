# -*- coding: UTF-8 -*-
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import conv
from theano.ifelse import ifelse

import tools
from test import test
import sys
import time
import random
import json
from collections import OrderedDict
import heapq

np.random.seed(970)

def ortho_weight(ndims):
    W = np.random.randn(ndims, ndims)
    u, s, v = np.linalg.svd(W)
    return u.astype(theano.config.floatX)

def numpy_floatX(data):
    return np.asarray(data, dtype=theano.config.floatX)

def initParams(Cemb,options):
    '''
        init params used in our model
    '''
    nhiddens = options['nhiddens']
    ndims = options['ndims']
    max_word_len = options['max_word_len']
    
    params = OrderedDict()
    params['Cemb'] = Cemb.astype(theano.config.floatX)
    lstmW = np.concatenate([ortho_weight(nhiddens),
                            ortho_weight(nhiddens),
                            ortho_weight(nhiddens),
                            ortho_weight(nhiddens)],axis=1)
    params['lstmW'] = lstmW
    lstmU = np.concatenate([ortho_weight(nhiddens),
                            ortho_weight(nhiddens),
                            ortho_weight(nhiddens),
                            ortho_weight(nhiddens)],axis=1)
    params['lstmU'] = lstmU
    lstmb = np.zeros((4*nhiddens,))
    for i in xrange(nhiddens,2*nhiddens):
        lstmb[i] = 1.
    params['lstmb'] = lstmb.astype(theano.config.floatX)

    presum = max_word_len*(max_word_len+1)/2
    params['rgW'] = np.asarray(np.random.uniform(low=-0.01,high=0.01,size=(max_word_len*nhiddens,presum*nhiddens))).astype(theano.config.floatX)
    params['rgb'] = np.zeros((presum*nhiddens,)).astype(theano.config.floatX)
    params['cW'] =np.asarray(np.random.uniform(low=-1.,high=1.,size=(nhiddens,presum*nhiddens))).astype(theano.config.floatX)
    
    past = 1
    for i in xrange(2,max_word_len+1):
        params['cW'][:,past*nhiddens:(past+i)*nhiddens]=  params['cW'][:,past*nhiddens:(past+i)*nhiddens]/float(i)
        past+=i
    params['cb'] =np.zeros((max_word_len*nhiddens,)).astype(theano.config.floatX)
    
    params['ugW'] = np.asarray(np.random.uniform(low=-0.01,high=0.01,size=((max_word_len+1)*nhiddens,(presum+max_word_len)*nhiddens))).astype(theano.config.floatX)
    params['ugb'] = np.zeros(((presum+max_word_len)*nhiddens,)).astype(theano.config.floatX)

    params['U'] = np.zeros((nhiddens,)).astype(theano.config.floatX)
    params['Wy'] = ortho_weight(nhiddens)
    params['by'] = np.zeros((nhiddens,)).astype(theano.config.floatX)
    params['<h>'] = np.zeros((nhiddens,)).astype(theano.config.floatX)
    return params

def initTparams(params):
    '''
        init Theano shared variables
    '''
    tparams = OrderedDict()
    for name,param in params.iteritems():
        tparams[name] = theano.shared(numpy_floatX(param), name=name)
    return tparams

def get_params(tparams):
    params = OrderedDict()
    for name,param in tparams.iteritems():
        params[name] = param.get_value(borrow=True)
    return params

def get_minibatches_idx(X,lens,batch_size,shuffle):
    n = len(X)
    idx_list = range(n)
    if shuffle:
        random.shuffle(idx_list)
    minibatches = []
    minibatch_start = 0
    idx_list = sorted(idx_list,cmp = lambda x,y: cmp(lens[x],lens[y]))
    for i in range(n//batch_size):
        minibatches.append(idx_list[minibatch_start:minibatch_start+batch_size])
        minibatch_start+=batch_size
    if minibatch_start!=n:
        minibatches.append(idx_list[minibatch_start:])
    if shuffle:
        random.shuffle(minibatches)
    return minibatches


def prepare_adadelta(tparams):
    ms_up = OrderedDict()
    ms_grad = OrderedDict()
    for kk,pp in tparams.iteritems():
        ms_up[kk] = theano.shared(np.zeros(pp.get_value().shape,dtype=theano.config.floatX),name = 'ms_up_%s'%kk)
        ms_grad[kk] = theano.shared(np.zeros(pp.get_value().shape,dtype=theano.config.floatX),name = 'ms_grad_%s'%kk)
    return ms_up,ms_grad

def adadelta(ms_up,ms_grad,tparams,cost,decay=0.95,epsilon=1e-6):
    updates =  OrderedDict()
    grads = T.grad(cost,tparams.values())
    for kk,grad in zip(tparams.keys(),grads):
        msu = ms_up[kk]
        msg = ms_grad[kk]
        updated_msg = decay*msg+(1-decay)*T.sqr(grad)
        updates[msg] = updated_msg
        step = -T.sqrt((msu+epsilon)/(updated_msg+epsilon))*grad
        updates[msu] = decay*msu+(1-decay)*T.sqr(step)
        updates[tparams[kk]] = tparams[kk]+step
    return updates

def prepare_adagrad(tparams):
    ss_grad = OrderedDict()
    for kk,pp in tparams.iteritems():
        ss_grad[kk] = theano.shared(np.zeros(pp.get_value().shape,dtype=theano.config.floatX),name = 'ss_grad_%s'%kk)
    return ss_grad

def adagrad(ss_grad,tparams,cost,lr,epsilon=1e-6):
    updates =  OrderedDict()
    grads = T.grad(cost,tparams.values())
    for kk,grad in zip(tparams.keys(),grads):
        updated_ss_grad = ss_grad[kk]+T.sqr(grad)
        step = -(lr/T.sqrt(updated_ss_grad+epsilon))*grad
        updates[ss_grad[kk]] = updated_ss_grad
        updates[tparams[kk]] = tparams[kk]+step
    return updates

def sgd(tparams,cost,lr):
    updates =  OrderedDict()
    grads = T.grad(cost,tparams.values())
    LR  = theano.shared(numpy_floatX(lr),name='LR')
    for kk,grad in zip(tparams.keys(),grads):
        step = -LR*grad
        updates[tparams[kk]] = tparams[kk]+step
    return LR,updates

def get_score(emb,seg,mask,tparams,options,_a_,_b_,_c_,_d_):
    nsamples = emb.shape[0]
    nsteps = emb.shape[1]
    ndims = emb.shape[2]
    nwords = seg.shape[0]
    nhiddens = options['nhiddens']
    def trick(xy):
        return (xy*(xy-1)/2)

    def fool_theano(true_len,true_start):
        virtual_len = ifelse(true_len,true_len,true_len+1)
        virtual_start = ifelse(true_len,true_start,true_start-1)
        return virtual_len,virtual_start
    
    def innerOneStep(idx,len,start,rgW,rgb,cW,cb,ugW,ugb,data):
        len,start = fool_theano(len,start)
        acc = trick(len)
        XL = data[idx,start:start+len,:].flatten()
        ln = len*nhiddens
        an = acc*nhiddens
        reset_gate=T.nnet.sigmoid(T.dot(rgW[:ln,an:an+ln],XL)+rgb[an:an+ln])
        com=T.concatenate([T.tanh(T.dot(cW[:,an:an+ln],reset_gate*XL)+cb[ln-nhiddens:ln]),XL])
        update_gate = T.exp(T.dot(ugW[:ln+nhiddens,an+ln-nhiddens:an+ln+ln],com)+ugb[an+ln-nhiddens:an+ln+ln]).reshape((len+1,nhiddens))
        word = (update_gate/update_gate.sum(axis=0))*(com.reshape((len+1,nhiddens)))
        word = word.sum(axis=0)
        return word
    
    def oneStep(word_len,msk,offset,h_,c_,score_,lstmW,lstmU,lstmb,Wy,by,U):
        idx_range = T.arange(nsamples)
        x,useless = theano.scan(
                    fn=innerOneStep,
                    sequences=[idx_range,word_len,offset],
                    non_sequences = [tparams['rgW'],
                                     tparams['rgb'],
                                     tparams['cW'],
                                     tparams['cb'],
                                     tparams['ugW'],
                                     tparams['ugb'],
                                     emb])
        preact = T.dot(x,lstmW)+T.dot(h_,lstmU)+lstmb
        i = T.nnet.sigmoid(preact[:,:nhiddens])
        f = T.nnet.sigmoid(preact[:,nhiddens:2*nhiddens])
        o = T.nnet.sigmoid(preact[:,2*nhiddens:3*nhiddens])
        c = T.tanh(preact[:,3*nhiddens:])
        c = f*c_+i*c
        h = o*T.tanh(c)
        y_ = T.tanh(T.dot(h_,Wy)+by)
        delta = T.dot(x,U)+(x*y_).sum(axis=1)
        score = score_+delta*msk
        new_offset = offset+word_len
        return new_offset,h,c,score
    
    results,updates = theano.scan(
            fn=oneStep,
            sequences=[seg,mask],
            outputs_info=[_a_[:nsamples],
                          _b_[:nsamples,:],
                          _c_[:nsamples,:],
                          _d_[:nsamples]],
            non_sequences = [tparams['lstmW'],
                             tparams['lstmU'],
                             tparams['lstmb'],
                             tparams['Wy'],
                             tparams['by'],
                             tparams['U']])
    return results[3][-1]

def build_model(tparams,options):
    x = T.matrix('x',dtype='int32')
    dropout = T.matrix('dropout',dtype=theano.config.floatX)
    y = T.matrix('y',dtype='int32')
    yy = T.matrix('yy',dtype='int32')
    y_mask = T.matrix('y_mask',dtype=theano.config.floatX)
    yy_mask = T.matrix('yy_mask',dtype=theano.config.floatX)
    
    #Performance
    max_size = (options['batch_size'],options['nhiddens'])
    _e_ = theano.shared(np.ones(max_size,dtype=theano.config.floatX))
    _c_ = theano.shared(np.zeros(max_size,dtype=theano.config.floatX))
    _a_ =theano.shared(np.zeros((options['batch_size'],),dtype='int32'))
    _d_ =theano.shared(np.zeros((options['batch_size'],),dtype=theano.config.floatX))
    _b_ = _e_*tparams['<h>']
    #Performance
    
    emb = tparams['Cemb'][x.flatten()].reshape((x.shape[0],x.shape[1],options['ndims']))
    emb = emb*dropout
    cost = get_score(emb,yy,yy_mask,tparams,options,_a_,_b_,_c_,_d_)-get_score(emb,y,y_mask,tparams,options,_a_,_b_,_c_,_d_)
    cost = T.cast(T.mean(cost),theano.config.floatX)
    return x,dropout,y,yy,y_mask,yy_mask,cost
    
def train_model(
    max_epochs = 30,
    optimizer = adadelta,
    batch_size = 256,
    ndims = 100,
    nhiddens = 150,
    dropout_rate = 0.,
    regularization = 0.,
    margin_loss_discount = 0.2,
    max_word_len = 4,
    start_point = 1,
    load_params = None,
    resume_training = False,
    max_sent_len = 60,
    beam_size = 4,
    shuffle_data = True,
    train_file = '../data/train',
    dev_file = '../data/dev',
    lr = 0.2,
    pre_training = '../w2v/c_vecs_100'
):
    options = locals().copy()
    print 'model options:',options
    print 'Building model'
    
    Cemb,character_idx_map = tools.initCemb(ndims,train_file,pre_training)
    
    print '%saving config file'
    config = {}
    config['options'] = options
    config['options']['optimizer'] = optimizer.__name__
    config['character_idx_map'] = character_idx_map
    f = open('config','wb')
    f.write(json.dumps(config))
    f.close()
    print '%resume model building'
    
    params = initParams(Cemb,options)
    if load_params is not None:
        pp = np.load(load_params)
        for kk,vv in params.iteritems():
            if kk not in pp:
                raise Warning('%s is not in the archive' % kk)
            params[kk] = pp[kk]
    tparams = initTparams(params)
    if optimizer is adadelta:
        ms_up,ms_grad = prepare_adadelta(tparams)
    if optimizer is adagrad:
        if resume_training:
            ss_grad = initTparams(np.load('backup.npz'))
        else:
            ss_grad = prepare_adagrad(tparams)
    T_x,T_dropout,T_y,T_yy,T_y_mask,T_yy_mask,T_cost = build_model(tparams,options)
    weight_decay = (tparams['U']**2).sum()+(tparams['Wy']**2).sum()
    weight_decay *= regularization
    T_cost += weight_decay

    if optimizer is adadelta:
        T_updates = optimizer(ms_up,ms_grad,tparams,T_cost)
    elif optimizer is sgd:
        LR,T_updates = optimizer(tparams,T_cost,lr)
    elif optimizer is adagrad:
        T_updates = optimizer(ss_grad,tparams,T_cost,lr)

    f_update = theano.function([T_x,T_dropout,T_y,T_yy,T_y_mask,T_yy_mask],T_cost,updates=T_updates)

    print 'Loading data'
    seqs,lenss,tagss = tools.prepareData(character_idx_map,train_file)
    if max_sent_len is not None:
        survived = []
        for idx,seq in enumerate(seqs):
            if len(seq)<=max_sent_len and len(seq)>1:
                survived.append(idx)
        seqs =  [ seqs[idx]  for idx in survived]
        lenss = [ lenss[idx] for idx in survived]
        tagss = [ tagss[idx] for idx in survived]

    tot_lens = [len(seq) for seq in seqs]
    print 'count_training_sentences',len(seqs)
    
    print 'Training model'
    start_time = time.time()
    for eidx in xrange(max_epochs):
        batches_idx = get_minibatches_idx(seqs,tot_lens,batch_size,shuffle=shuffle_data)
        for batch_idx in batches_idx:
            X = [seqs[t]  for t in batch_idx]
            Y = [lenss[t] for t in batch_idx]
            Z = [tagss[t] for t in batch_idx]
            X_lens = [tot_lens[t] for t in batch_idx]
            params = get_params(tparams)
            X = tools.asMatrix(X)
            dropout = np.random.binomial(1,1-dropout_rate,(X.shape[1],ndims)).astype(theano.config.floatX)
            #numpy_start = time.time()
            YY= tools.segment(params,options,X,X_lens,dropout,margin_loss_discount,Z)
            #print 'numpy',time.time()-numpy_start
            Y = tools.asMatrix(Y,transpose=True)
            YY = tools.asMatrix(YY,transpose=True)
            Y_mask = (Y/Y).astype(theano.config.floatX)
            YY_mask =(YY/YY).astype(theano.config.floatX)
            #theano_start = time.time()
            f_update(X,dropout,Y,YY,Y_mask,YY_mask)
            #print 'theano',time.time()-theano_start
        if optimizer is sgd:
            LR.set_value(numpy_floatX(LR.get_value()*0.9))
        params = get_params(tparams)
        test(config['character_idx_map'],config['options'],params,dev_file,'../result/dev_result%s'%(eidx+start_point,))
        np.savez('epoch_%s'%(eidx+start_point,),**params)
        if optimizer is adagrad:
            np.savez('backup',**get_params(ss_grad))
        end_time = time.time()
        print 'Trained %s epoch(s) took %.lfs per epoch'%(eidx+1,(end_time-start_time)/(eidx+1))

