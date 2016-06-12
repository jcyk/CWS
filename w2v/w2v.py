# -*- coding: UTF-8 -*-
import gensim
import re
import numpy as np
from collections import defaultdict

def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:                              #全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248
        
        rstring += unichr(inside_code)
    return rstring

class MySentences(object):
    def __init__(self,filename):
        self.filename = filename
    def __iter__(self):
        for line in open(self.filename):
            yield [x.encode('utf8') for x in list(line.decode('utf8').strip())]

f= open('corpora')
fo= open('wiki','wb')
for line in f.readlines():
    sent = strQ2B(unicode(line.decode('utf8')).strip())
    if len(sent)>0:
        fo.write(sent.encode('utf8')+'\n')
f.close()
fo.close()

sents = MySentences('wiki')
sizes = [50,60,70,80,90,100]
for s in sizes:
    model = gensim.models.Word2Vec(sents,size=s,window=8,workers=4,max_vocab_size=10000,iter=5)
    model.save('c_vecs_%s'%(s,))

