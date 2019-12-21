''' author: samtenka
    change: 2019-12-21
    create: 2019-12-20
    descrp: fully connected classifier on MNIST 
    to use: run `python fully_connected.py`.
'''

import tensorflow as tf
import numpy as np

#=============================================================================#
#    0. LIST PROGRAM PARAMETERS                                               #
#=============================================================================#

# 0.0. Dataset Sizes:
TRAIN_SIZE = 100
TEST_SIZE  = 100

# 0.1. (Hyper)Parameters of Stochastic Gradient Descent:
NB_UPDATES = 2000
LACONICITY =  200  
BATCH_SIZE =  100
LEARN_RATE = 10.0

#=============================================================================#
#    1. OBTAIN DATASET                                                        #
#=============================================================================#

import boolean_data
from boolean_data import nb_feats, nb_texts, nb_tbins

# 1.0. Generate data
print('generating data...')
X, T, Y = boolean_data.generate(TRAIN_SIZE+TEST_SIZE)
X, T, Y = boolean_data.featurize(X, T, Y)
print('~ %.1f%% of files are buggy' % (np.mean(Y)*100))

# 1.1. Define data sampler 
def get_batch(size=BATCH_SIZE):
    ''' Return `inputs` of shape (28*28,) and the corresponding
               `outputs` of shape (10,)
        randomly sampled from the full data. 
    '''
    indices = (
        np.arange(TRAIN_SIZE, TRAIN_SIZE+TEST_SIZE)
        if size == 'test' else
        np.random.choice(TRAIN_SIZE, size=size, replace=False)
    )
    return X[indices], T[indices], Y[indices]

#=============================================================================#
#    2. BUILD COMPUTATION GRAPH                                               #
#=============================================================================#

def declare_var(name, shape):
    return tf.get_variable(
        name, shape, dtype=tf.float32, initializer=tf.zeros_initializer()
    )

# 2.0. Our current estimate of the true weights:
W_xy = declare_var('wxy', shape=[nb_feats,  nb_texts])
W_ty = declare_var('wty', shape=[nb_tbins-1,nb_texts])
B_y  = declare_var('by',  shape=[           nb_texts])
#weights = [W_xy, W_ty, B_y]

# 2.1. Placeholders for the data to which to fit `Weights` and `Biases`.  Note
#      that both `TrueInputs` and `TrueOutputs` are inputs to our graph:
TrueX = tf.placeholder(tf.float32, shape=[None, nb_feats])
TrueT = tf.placeholder(tf.float32, shape=[None, nb_tbins-1])
TrueY = tf.placeholder(tf.float32, shape=[None, nb_texts])
Threshold = tf.placeholder(tf.float32)  

# 2.2. This line is our mathematical regression model.  Here, it is logistic.
#      In practice, one often uses more complicated multistatement expressions.  
#      Note: for numerical stability, we clip the values away from 0 and 1.
PredictedProb = tf.clip_by_value(1.0/(1.0+tf.exp(-(
    tf.matmul(TrueX, W_xy) + 
    tf.matmul(TrueT, W_ty) +
    B_y 
))), clip_value_min=0.0001, clip_value_max=0.9999)

# 2.3. Gradient descent acts to minimize a differentiable loss (here, the Cross
#      Entropy Loss combined with an Regularizer):
EntropyLoss = -tf.reduce_mean(
    tf.multiply(    TrueY, tf.log(    PredictedProb)) +  
    tf.multiply(1.0-TrueY, tf.log(1.0-PredictedProb))
)

L1Regularizer = (
    0.01 * tf.reduce_mean(tf.abs(W_xy)) + 
    0.01 * tf.reduce_mean(tf.abs(W_ty)) + 
    0.01 * tf.reduce_mean(tf.abs(B_y ))   
)

EntLossGradNorm = 0#tf.reduce_mean(tf.abs(tf.gradients(EntropyLoss, weights)))
L1RegGradNorm = 0#tf.reduce_mean(tf.abs(tf.gradients(L1Regularizer, weights)))

Loss = EntropyLoss + L1Regularizer

# 2.4. Gradient descent step:
LearnRate = tf.placeholder(dtype=tf.float32)
Update = tf.train.GradientDescentOptimizer(LearnRate).minimize(Loss)

# 2.5. Classification diagnostics (how well did we do?).
Prediction = tf.math.round(PredictedProb - Threshold + 0.5)

Recall = (
    tf.reduce_sum(tf.multiply(Prediction, TrueY)) /
    tf.reduce_sum(                        TrueY )  
)

Precision = (
    tf.reduce_sum(tf.multiply(Prediction, TrueY)) /
    tf.reduce_sum(            Prediction        )  
)

#=============================================================================#
#    3. RUN GRAPH                                                             #
#=============================================================================#

# 3.0. 
def report_accuracy(sess, i):
    ''' print precision, recall, and optimization objectives on test set '''
    batch_x, batch_t, batch_y = get_batch('test') 
    precision, recall, loss, reg = sess.run(
        [Precision, Recall, EntropyLoss, L1Regularizer],
        feed_dict={
            TrueX:batch_x, TrueT:batch_t, TrueY:batch_y, Threshold:0.2 
        }
    ) 
    print('    '.join((
        'iter %4d'       % i,
        'precision %.2f' % precision,
        'recall %.2f'    % recall,
        'loss %.3f'      % loss,
        'reg %.1e'       % reg,      
        #'lossg %.2e'     % lossg,
        #'regg %.2e'      % regg,
    )))

# 3.1.
def learn(sess): 
    ''' perform one gradient update based on a batch of examples '''
    batch_x, batch_t, batch_y = get_batch() 
    sess.run(Update,
        feed_dict={
            TrueX:batch_x, TrueT:batch_t, TrueY:batch_y, LearnRate:LEARN_RATE
        }
    )

# 3.2. Repeatedly learn, every so often reporting accuracy: 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(NB_UPDATES+1):
        if i % LACONICITY==0:
            report_accuracy(sess, i)
        learn(sess)

