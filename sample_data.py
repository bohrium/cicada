''' author: samtenka
    change: 2019-12-20
    create: 2019-12-20
    descrp: fully connected classifier on MNIST 
    to use: Run `python fully_connected.py`.
'''

import tensorflow as tf
import numpy as np

###############################################################################
#                         0. LIST PROGRAM PARAMETERS                          #
###############################################################################

# 0.0. (Hyper)Parameters of Stochastic Gradient Descent:

nb_feats = 24
time_min = 0.0 
time_max = 5.0
nb_texts = 100
nb_tbins = 5

def sigmoid(arr):
    return 1.0 / (1.0+np.exp(-arr))

w_m = np.random.normal(size=(nb_feats, nb_texts)) / (time_max-time_min)
w_b = np.random.normal(size=(nb_feats, nb_texts))
def generate(nb_samples, threshold=0.999):
    ''' basic generative model: t doesn't matter '''
    x = np.random.choice(2, size=(nb_samples, nb_feats)) 
    t = np.random.uniform(size=(nb_samples, 1)) * (time_max-time_min) + time_min
    p = sigmoid(np.matmul(t*x, w_m) + np.matmul(x, w_b))
    y = (threshold<p).astype(np.int32)
    return (x, t, y)

def featurize(x, t, y):
    ''' the number of features is one less than the number of bins '''
    time_bins = np.arange(time_min, time_max, (time_max-time_min)/nb_tbins)
    ts = np.repeat(t, repeats=nb_tbins-1, axis=1)
    t_binned = (ts < np.expand_dims(time_bins[1:], axis=0))
    return x.astype(np.float32), t_binned.astype(np.float32), y

N = 10000
X, T, Y = generate(N)
print('~ %.3f bugs' % (np.sum(Y)/(N*nb_texts)))

