import os
import sys

type = sys.argv[1]
s = int(sys.argv[2])
e = int(sys.argv[3])

for i in xrange(s,e+1):
    cmd = './score ../data/dic ../data/%s_test ../result/dev_result%s > tmp'%(type,i)
    os.system(cmd)
    cmd = 'grep \'F MEASURE\' tmp '
    os.system(cmd)
    cmd = 'rm tmp'
    os.system(cmd)

