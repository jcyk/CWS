# -*- coding: UTF-8 -*-
import re

Maximum_Word_Length = 4 
pku_train_file ='../original/pku_training.utf8'
pku_test_file = '../original/pku_test_gold.utf8'
msr_train_file ='../original/msr_training.utf8'
msr_test_file = '../original/msr_test_gold.utf8'
pku_longwords = '../data/pku_longwords' # one should generate the file first using preprocess's output
msr_longwords = '../data/msr_longwords'# one should generate the file first using preprocess's output
chinese_idioms_file = '../data/idioms'

def OT(str):
    print str.encode('utf8')

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

def preprocess(path,filename,longwords=None):
    idioms = []
    f = open(chinese_idioms_file)
    for line in f.readlines():
        idiom = unicode(line.decode('utf8')).strip()
        idioms.append(idiom)
    f.close()

    longws = []
    if longwords is not None:
        f = open(longwords)
        for line in f.readlines():
            longword = unicode(line.decode('utf8')).strip()
            longws.append(longword)
        f.close()
    longws = set(longws)
    idioms = set(idioms)
    count_idioms = 0
    count_longws = 0
    f = open(path)
    sents = []
    rNUM = u'(-|\+)?\d+((\.|·)\d+)?%?'
    rENG = u'[A-Za-z_.]+'
    for line in f.readlines():
        sent = strQ2B(unicode(line.decode('utf8')).strip()).split(u'  ')
        new_sent = []
        for word in sent:
            word = re.sub(u'\s+','',word,flags =re.U)
            word = re.sub(rNUM,u'0',word,flags= re.U)
            word = re.sub(rENG,u'X',word)
            #if word in idioms:      # use Chinese idioms dictionary.
            #   count_idioms+=1
            #    word = u'I'
            if word in longws:       #to remove those word longer than our maximum word length setting.
                count_longws+=1
                word = u'L'
            new_sent.append(word)
        sents.append(new_sent)
    f.close()
    print 'idioms count',count_idioms
    print 'long words count',count_longws
    check(sents)
    f= open(filename,'wb')
    for sent in sents:
        f.write('  '.join(sent).encode('utf8')+'\r\n')
    f.close()


def check(sents): # get those words longer than our maximum word length setting
    count = 0
    all_count = 0
    longwords = []
    for sent in sents:
        for word in sent:
            all_count+=1
            if len(word)>Maximum_Word_Length:
                count+=1
                longwords.append(word)
    for word in set(longwords):
        OT(word)
    print 'len>%d words count'%Maximum_Word_Length,count,100.0*count/all_count,'%'

preprocess(pku_test_file,'../data/pku_test')
