''' author: samtenka
    change: 2019-12-20
    create: 2019-12-20
    descrp: toy model of bugs pattern in a code base
    to use: import
'''

import tensorflow as tf
import numpy as np

nb_feats = 24
time_min = 0.0 
time_max = 5.0
nb_texts = 100
nb_tbins = 12

def random_mask(nb_conj = 8, nb_disj = 2):
    feat = lambda : (
        ('not ' if np.random.uniform() < 0.5 else '') + 
        ('x[%d]' % np.random.choice(nb_feats)
         if np.random.uniform() < 0.8 else
         '(%.2f<=t)' % (np.random.uniform()*(time_max-time_min) + time_min)
        )
    )
    term = lambda : ( 
        ' and '.join(feat() for i in range(nb_conj))
    )
    return ( 
        ' or '.join(term() for i in range(nb_disj))
    )

def eval_mask(mask_str, xx, tt):
    return eval(mask_str, {'x':xx, 't':tt})

masks = [random_mask() for t in range(nb_texts)]

def generate(nb_samples):
    ''' basic generative model: t doesn't matter '''
    x = np.random.choice(2, size=(nb_samples, nb_feats)) 
    t = np.random.uniform(size=(nb_samples, 1)) * (time_max-time_min) + time_min
    y = np.array([[eval_mask(masks[i], x[s], t[s]) for i in range(nb_texts)] for s in range(nb_samples)])
    return (x, t, y)

def featurize(x, t, y):
    ''' the number of features is one less than the number of bins '''
    time_bins = np.arange(time_min, time_max, (time_max-time_min)/nb_tbins)
    ts = np.repeat(t, repeats=nb_tbins-1, axis=1)
    t_binned = (ts < np.expand_dims(time_bins[1:], axis=0))
    return x.astype(np.float32), t_binned.astype(np.float32), y

