#!/usr/bin/env python
# coding: utf-8
# %%
import os
from multiprocessing import Pool
import sys
_round=sys.argv[1]
# _round=1


# %%


def fun(number):
    sh_file = 'gpu' + str(number) + '.sh ' + str(_round)
    os.system('sh ' + sh_file)


# %%


POOL_SIZE = 8
pool = Pool(POOL_SIZE)


# %%


l=[0,1,2,3,4,5,6,7]


# %%


pool.map(fun,l)
pool.close()
pool.join()
del pool


# %%




