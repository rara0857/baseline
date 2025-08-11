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


POOL_SIZE = 1
pool = Pool(POOL_SIZE)


# %%


l=[0]


# %%


pool.map(fun,l)
pool.close()
pool.join()
del pool


# %%




