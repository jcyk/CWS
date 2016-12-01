# -*- coding: UTF-8 -*-
import dynet as dy
from collections import namedtuple
from tools import initCemb,prepareData
from model import get_minibatches_idx
from dy_test import test
import numpy as np
import random,time

Sentence = namedtuple('Sentence',['score','score_expr','LSTMState','y','prevState','wlen'])

class Agenda(object): # maintain the top K hightest scored partially segmented sentences
    def __init__(self,beam_size):
        self.size = 0
        self.beam_size = beam_size
        self.data = [None] * beam_size

    def pop(self, sent):
        pos = 0
        while 2*pos+1 < self.beam_size:
            if 2*pos+2 >= self.beam_size or self.data[2*pos+1].score < self.data[2*pos+2].score:
                self.data[pos] = self.data[2*pos+1]
                pos = 2*pos+1
            else:
                self.data[pos] = self.data[2*pos+2]
                pos = 2*pos+2
        self.data[pos] = sent
        self.adjust(pos)

    def adjust(self,pos):
        while pos > 0 and self.data[(pos-1)/2].score > self.data[pos].score:
            next_pos = (pos-1)/2
            self.data[pos],self.data[next_pos] = self.data[next_pos],self.data[pos]
            pos = next_pos

    def push(self, sent):
        if self.size < self.beam_size:
            self.data[self.size] = sent
            self.adjust(self.size)
            self.size+=1
        else:
            self.pop(sent)

    def max(self):
        return max(self.data[:self.size],key=lambda x: x.score)

    def happy_with(self, score):
        if self.size < self.beam_size:
            return True
        else:
            return score > self.data[0].score

    def __iter__(self):
        return (self.data[:self.size]).__iter__()

     

class CWS (object):
    def __init__(self,Cemb,character_idx_map,options):
        model = dy.Model()
        self.trainer = dy.AdagradTrainer(model,options['lr']) # we use Adagrad
        self.params = self.initParams(model,Cemb,options)
        self.options = options
        self.model = model
        self.character_idx_map = character_idx_map
    
    def load(self,filename):
        self.model.load(filename)

    def save(self,filename):
        self.model.save(filename)

    def initParams(self,model,Cemb,options):
        # initialize the model parameters  
        params = dict()
        params['embed'] = model.add_lookup_parameters(Cemb.shape)
        for row_num,vec in enumerate(Cemb):
            params['embed'].init_row(row_num, vec)
        params['lstm'] = dy.LSTMBuilder(1,options['ndims'],options['nhiddens'],model)
        
        params['reset_gate_W'] = []
        params['reset_gate_b'] = []
        params['com_W'] = []
        params['com_b'] = []
        params['update_gate_W'] = []
        params['update_gate_b'] = []
        
        params['word_score_U'] = model.add_parameters(options['ndims'])
        params['predict_W'] = model.add_parameters((options['ndims'],options['nhiddens']))
        params['predict_b'] = model.add_parameters(options['ndims'])
        for wlen in xrange(1,options['max_word_len']+1):
            params['reset_gate_W'].append(model.add_parameters((wlen*options['ndims'],wlen*options['ndims'])))
            params['reset_gate_b'].append(model.add_parameters(wlen*options['ndims']))
            params['com_W'].append(model.add_parameters((options['ndims'],wlen*options['ndims'])))
            params['com_b'].append(model.add_parameters(options['ndims']))
            params['update_gate_W'].append(model.add_parameters(((wlen+1)*options['ndims'],(wlen+1)*options['ndims'])))
            params['update_gate_b'].append(model.add_parameters( (wlen+1)*options['ndims']))
        params['<BoS>'] = model.add_parameters(options['ndims'])
        return params
    
    def renew_cg(self):
        # renew the compute graph for every single instance
        dy.renew_cg()

        param_exprs = dict()
        param_exprs['U'] = dy.parameter(self.params['word_score_U'])
        param_exprs['pW'] = dy.parameter(self.params['predict_W'])
        param_exprs['pb'] = dy.parameter(self.params['predict_b'])
        param_exprs['<bos>'] = dy.parameter(self.params['<BoS>'])
        self.param_exprs = param_exprs
    
    def word_repr(self, char_seq):
        # obtain the word representation when given its character sequence
        wlen = len(char_seq)
        if 'rgW%d'%wlen not in self.param_exprs:
            self.param_exprs['rgW%d'%wlen] = dy.parameter(self.params['reset_gate_W'][wlen-1])
            self.param_exprs['rgb%d'%wlen] = dy.parameter(self.params['reset_gate_b'][wlen-1])
            self.param_exprs['cW%d'%wlen] = dy.parameter(self.params['com_W'][wlen-1])
            self.param_exprs['cb%d'%wlen] = dy.parameter(self.params['com_b'][wlen-1])
            self.param_exprs['ugW%d'%wlen] = dy.parameter(self.params['update_gate_W'][wlen-1])
            self.param_exprs['ugb%d'%wlen] = dy.parameter(self.params['update_gate_b'][wlen-1])
          
        chars = dy.concatenate(char_seq)
        reset_gate = dy.logistic(self.param_exprs['rgW%d'%wlen] * chars + self.param_exprs['rgb%d'%wlen])
        comb = dy.concatenate([dy.tanh(self.param_exprs['cW%d'%wlen] * dy.cmult(reset_gate,chars) + self.param_exprs['cb%d'%wlen]),chars])
        update_logits = self.param_exprs['ugW%d'%wlen] * comb + self.param_exprs['ugb%d'%wlen]
        
        update_gate = dy.transpose(dy.concatenate_cols([dy.softmax(dy.pickrange(update_logits,i*(wlen+1),(i+1)*(wlen+1))) for i in xrange(self.options['ndims'])]))
        
        # The following implementation of Softmax fucntion is not safe, but faster...
        #exp_update_logits = dy.exp(dy.reshape(update_logits,(self.options['ndims'],wlen+1)))
        #update_gate = dy.cdiv(exp_update_logits, dy.concatenate_cols([dy.sum_cols(exp_update_logits)] *(wlen+1)))
        #assert (not np.isnan(update_gate.npvalue()).any())

        word = dy.sum_cols(dy.cmult(update_gate,dy.reshape(comb,(self.options['ndims'],wlen+1))))
        return word

    def beam_search(self, char_seq, truth = None, mu =0.): 
        start_agenda = Agenda(self.options['beam_size'])
        init_state = self.params['lstm'].initial_state().add_input(self.param_exprs['<bos>'])
        init_y = dy.tanh(self.param_exprs['pW'] * init_state.output() + self.param_exprs['pb'])
        init_score = dy.scalarInput(0.)
        start_agenda.push(Sentence(score=init_score.scalar_value(),score_expr=init_score,LSTMState =init_state, y= init_y , prevState = None, wlen=None))
        agenda = [start_agenda]

        for idx, _ in enumerate(char_seq,1): # from left to right, character by character
            now = Agenda(self.options['beam_size'])
            for wlen in xrange(1,min(idx,self.options['max_word_len'])+1): # generate candidate word vectors
                word = self.word_repr(char_seq[idx-wlen:idx])
                word_score = dy.dot_product(word,self.param_exprs['U'])
                for sent in agenda[idx-wlen]: # join segmentation
                    if truth is not None:
                        margin = dy.scalarInput(mu*wlen if truth[idx-1]!=wlen else 0.)
                        score = margin + sent.score_expr + dy.dot_product(sent.y, word) + word_score 
                    else:
                        score = sent.score_expr + dy.dot_product(sent.y, word) + word_score 
                    
                    if now.happy_with(score.scalar_value()):
                        new_state = sent.LSTMState.add_input(word)
                        new_y = dy.tanh(self.param_exprs['pW'] * new_state.output() + self.param_exprs['pb'])
                        now.push(Sentence(score=score.scalar_value(),score_expr=score,LSTMState=new_state,y=new_y, prevState=sent, wlen=wlen))
            agenda.append(now)

        if truth is not None:
            return agenda[-1].max().score_expr
        return agenda

    def truth_score(self, word_seq):

        wembs = [self.param_exprs['<bos>']]+[self.word_repr(word) for word in word_seq]
        init_state = self.params['lstm'].initial_state()
        hidden_states = init_state.transduce(wembs)
        score = dy.scalarInput(0.)
        for h, w in zip(hidden_states[:-1],wembs[1:]):
            y = dy.tanh(self.param_exprs['pW'] * h + self.param_exprs['pb'])
            score = score + dy.dot_product(y,w) +dy.dot_product(w,self.param_exprs['U']) 
        return score

    def forward(self, char_seq):
        self.renew_cg()
        cembs = [dy.lookup(self.params['embed'],char) for char in char_seq ]
        
        agenda = self.beam_search(cembs)
        now = agenda[-1].max()
        ans = []
        while now.prevState is not None:
            ans.append(now.wlen)
            now = now.prevState
        return reversed(ans)


    def backward(self, char_seq, truth):
        self.renew_cg()

        cembs = [ dy.dropout(dy.lookup(self.params['embed'],char),self.options['dropout_rate']) for char in char_seq ]
    
        word_seq,word = [],[]
        for char,label in zip(cembs,truth):
            word.append(char)
            if label > 0:
                word_seq.append(word)
                word = []

        score = self.truth_score(word_seq)

        score_plus_margin_loss = self.beam_search(cembs,truth,self.options['margin_loss_discount'])

        loss = score_plus_margin_loss - score

        res = loss.scalar_value()
        loss.backward()
        return res

def dy_train_model(
    max_epochs = 50,
    batch_size = 256,
    ndims = 50,
    nhiddens = 50,
    dropout_rate = 0.2,
    regularization = 0.000001,
    margin_loss_discount = 0.2,
    max_word_len = 4,
    start_point = 1,
    load_params = None,
    max_sent_len = 60,
    beam_size = 4,
    shuffle_data = True,
    train_file = '../data/train',
    dev_file = '../data/dev',
    lr = 0.2,
    pre_training = '../w2v/c_vecs_50'
):
    options = locals().copy()
    print 'Model options:'
    for kk,vv in options.iteritems():
        print '\t',kk,'\t',vv
    
    Cemb, character_idx_map = initCemb(ndims,train_file,pre_training)

    cws = CWS(Cemb,character_idx_map,options)

    if load_params is not None:
        cws.load(load_params)

    char_seq, _ , truth = prepareData(character_idx_map,train_file)
    
    if max_sent_len is not None:
        survived = []
        for idx,seq in enumerate(char_seq):
            if len(seq)<=max_sent_len and len(seq)>1:
                survived.append(idx)
        char_seq =  [ char_seq[idx]  for idx in survived]
        truth = [ truth[idx] for idx in survived]
    n = len(char_seq)
    print 'Total number of training instances:',n
    
    print 'Start training model'
    start_time = time.time()
    nsamples = 0
    for eidx in xrange(max_epochs):
        
        idx_list = range(n)
        if shuffle_data:
            random.shuffle(idx_list)

        for idx in idx_list:
            loss = cws.backward(char_seq[idx],truth[idx])
            if np.isnan(loss):
                print 'somthing went wrong, loss is nan.'
                return
            nsamples += 1
            if nsamples % batch_size == 0:
                cws.trainer.update(1./batch_size)

        cws.trainer.update_epoch(1.)
        end_time = time.time()
        print 'Trained %s epoch(s) (%d samples) took %.lfs per epoch'%(eidx+1,nsamples,(end_time-start_time)/(eidx+1))       
        test(cws,dev_file,'../result/dev_result%d'%(eidx+start_point))
        #cws.save('epoch%d'%(eidx+start_point))
        #print 'Current model saved'
