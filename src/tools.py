# -*- coding: UTF-8 -*-
from collections import defaultdict
from collections import namedtuple
from collections import OrderedDict

import numpy as np
import theano
import random
import gensim
import scipy
import re
import heapq

def initCemb(ndims,train_file,pre_trained,thr = 5.):
    f = open(train_file)
    train_vocab = defaultdict(float)
    for line in f.readlines():
        sent = unicode(line.decode('utf8')).split()
        for word in sent:
            for character in word:
                train_vocab[character]+=1
    f.close()
    character_vecs = {}
    if pre_trained is not None:
        pre_trained = gensim.models.Word2Vec.load(pre_trained)
        pre_trained_vocab = set([ unicode(w.decode('utf8')) for w in pre_trained.vocab.keys()])
    for character in train_vocab:
        if train_vocab[character]< thr:
            continue
        character_vecs[character] = np.random.uniform(-0.5,0.5,ndims)
    for character in pre_trained_vocab:
        character_vecs[character] = pre_trained[character.encode('utf8')]
    Cemb = np.zeros(shape=(len(character_vecs)+1,ndims))
    idx = 1
    character_idx_map = dict()
    for character in character_vecs:
        Cemb[idx] = character_vecs[character]
        character_idx_map[character] = idx
        idx+=1
    return Cemb,character_idx_map

def SMEB(lens):
    idxs = []
    for len in lens:
        for i in xrange(len-1):
            idxs.append(0)
        idxs.append(len)
    return idxs

def prepareData(character_idx_map,path,test=False):
    seqs,wlenss,idxss = [],[],[]
    f = open(path)
    for line in f.readlines():
        sent = unicode(line.decode('utf8')).split()
        Left = 0
        for idx,word in enumerate(sent):
            if len(re.sub('\W','',word,flags=re.U))==0:
                if idx >Left:
                    seqs.append(list(''.join(sent[Left:idx])))
                    wlenss.append([len(word) for word in sent[Left:idx]])
                Left = idx+1
        if Left!=len(sent):
            seqs.append(list(''.join(sent[Left:])))
            wlenss.append([len(word) for word in sent[Left:]])
    seqs = [[  character_idx_map[character] if character in character_idx_map else 0 for character in seq] for seq in seqs]
    f.close()
    if test:
        return seqs
    for wlens in wlenss:
        idxss.append(SMEB(wlens))
    return seqs,wlenss,idxss

def sigmoid (x):
    return 1.0/(1.0+np.exp(-x))

def tanh(x):
    return np.tanh(x)

def asMatrix(X,transpose=False):
    nsamples = len(X)
    lens = [len(x) for x in X]
    max_len  = max(lens)
    matrix = np.zeros((nsamples,max_len)).astype('int32')
    for idx,x in enumerate(X):
        matrix[idx,:lens[idx]] = x
    if transpose:
        matrix = np.transpose(matrix)
    return matrix

LSTMState = namedtuple('LSTMState',['score','margin_loss','randomValue','prevState','h','c','len'])

class TopkHeap(object):
    def __init__(self,k):
        self.k = k
        self.data = []
    def push(self,x):
        if self.k<0 or len(self.data)<self.k:
            heapq.heappush(self.data,x)
        else:
            topk_small = self.data[0]
            if x > topk_small:
                heapq.heapreplace(self.data,x)
    def Topk(self):
        return self.data
    def max(self):
        return max(self.data)

class LSTMCell(object):
    def __init__(self,params,nhiddens):
        self.W = params['lstmW']
        self.U = params['lstmU']
        self.b = params['lstmb']
        self.h0 = params['<h>']
        self.c0 = np.zeros((nhiddens,)).astype(theano.config.floatX)
        self.dim =nhiddens
    
    def ready(self):
        return self.h0,self.c0
    
    def step(self,h_,c_,x):
        
        def _slice(matrix,n):
            return matrix[n*self.dim:(n+1)*self.dim]
        
        preact = np.dot(x,self.W)+np.dot(h_,self.U)+self.b
        i = sigmoid(_slice(preact, 0))
        f = sigmoid(_slice(preact, 1))
        o = sigmoid(_slice(preact, 2))
        c = tanh(_slice(preact, 3))
        c = f*c_+i*c
        h = o*np.tanh(c)
        return h,c

def get_word(ndims,len,XL,rgW,rgb,cW,cb,ugW,ugb):
    ln = len*ndims
    an = len*(len-1)*ndims/2
    reset_gate=sigmoid(np.dot(rgW[:ln,an:an+ln],XL)+rgb[an:an+ln])
    com=np.concatenate([tanh(np.dot(cW[:,an:an+ln],reset_gate*XL)+cb[ln-ndims:ln]),XL])
    update_gate = np.exp(np.dot(ugW[:ln+ndims,an+ln-ndims:an+ln+ln],com)+ugb[an+ln-ndims:an+ln+ln]).reshape((len+1,ndims))
    word = (update_gate/update_gate.sum(axis=0))*(com.reshape((len+1,ndims)))
    word = word.sum(axis=0)
    return word

def segment(params,options,X,X_lens,dropout,discount=0.,Z=None):
    '''
        discount=0. and Z = None for test
    '''
    
    nsamples,nsteps,ndims,nhiddens =X.shape[0],X.shape[1],options['ndims'],options['nhiddens']
    max_width = min(options['max_word_len'],nsteps)
    lstm = LSTMCell(params,nhiddens)
    Wy = params['Wy']
    by = params['by']
    
    emb = params['Cemb'][X.flatten()].reshape((nsamples,nsteps,ndims))
    emb = emb*dropout
    Y = []
    beam_size=options['beam_size']
    sampleId = -1
    for x in X:
        sampleId+=1
        h,c = lstm.ready()
        now = TopkHeap(beam_size)
        now.push(LSTMState(score=0.,margin_loss=0.,h=h,c=c,prevState=None,len=None,randomValue = random.random()))
        history = [now]
        for t in xrange(1,1+X_lens[sampleId]):
            now = TopkHeap(beam_size)
            for prev in xrange(max(0,t-max_width),t):
                len = t-prev
                word = get_word(ndims,
                                len,
                                emb[sampleId,prev:prev+len,:].flatten(),
                                params['rgW'],
                                params['rgb'],
                                params['cW'],
                                params['cb'],
                                params['ugW'],
                                params['ugb'])
                states = history[prev].Topk()
                for state in states:
                    h,c = lstm.step(state.h,state.c,word)
                    y = tanh(np.dot(state.h,Wy)+by)
                    score = state.score+np.dot(word,params['U'])+np.dot(y,word)
                    delta = 0.
                    if Z is not None:
                        delta= (discount*len if Z[sampleId][t-1]!=len else 0.)
                        score+=delta
                    now.push(LSTMState(score=score,margin_loss=delta,h=h,c=c,prevState=state,len=len,randomValue=random.random()))
            history.append(now)
        back = history[-1].max()
        result = []
        while back.prevState is not None:
            result.append(back.len)
            back = back.prevState
        Y.append([len for len in reversed(result)])
    return Y
