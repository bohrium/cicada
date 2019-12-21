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
NB_UPDATES = 2000
TRAIN_SIZE = 1000
TEST_SIZE  = 1000
BATCH_SIZE =  100
LEARN_RATE = 10.0

###############################################################################
#                            1. READ DATASET                                  #
###############################################################################

import toy_data
from toy_data import nb_feats, nb_texts, nb_tbins

X, T, Y = toy_data.generate(TRAIN_SIZE+TEST_SIZE)
X, T, Y = toy_data.featurize(X, T, Y)

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

###############################################################################
#                         2. BUILD COMPUTATION GRAPH                          #
###############################################################################

def declare_var(name, shape):
    return tf.get_variable(
        name, shape, dtype=tf.float32, initializer=tf.zeros_initializer()
    )

# 2.0. Our current estimate of the true weights:
W_xy = declare_var('wxy', shape=[nb_feats,  nb_texts])
W_ty = declare_var('wty', shape=[nb_tbins-1,nb_texts])
B_y  = declare_var('by',  shape=[           nb_texts])

Regularizer = (
    0.01 * tf.reduce_mean(tf.abs(W_xy)) + 
    0.01 * tf.reduce_mean(tf.abs(W_ty)) + 
    0.01 * tf.reduce_mean(tf.abs(B_y ))   
)

# 2.1. Placeholders for the data to which to fit `Weights` and `Biases`.  Note
#      that both `TrueInputs` and `TrueOutputs` are inputs to our graph:
TrueX = tf.placeholder(tf.float32, shape=[None, nb_feats])
TrueT = tf.placeholder(tf.float32, shape=[None, nb_tbins-1])
TrueY = tf.placeholder(tf.float32, shape=[None, nb_texts])
Threshold = tf.placeholder(tf.float32)  

# 2.2. This line is our mathematical regression model.  Here, it is logistic.
#      In practice, one often uses more complicated multistatement expressions.  
PredictedProb = tf.clip_by_value(1.0/(1.0+tf.exp(-(
    tf.matmul(TrueX, W_xy) + 
    tf.matmul(TrueT, W_ty) +
    B_y 
))), clip_value_min=0.0001, clip_value_max=0.9999)

# 2.3. Gradient Descent acts to minimize a differentiable loss
#      (here, the Cross Entropy Loss):
EntropyLoss = -tf.reduce_mean(
    tf.multiply(    TrueY, tf.log(    PredictedProb)) +  
    tf.multiply(1.0-TrueY, tf.log(1.0-PredictedProb))
)

# 2.4. Gradient Descent Step:
LearnRate = tf.placeholder(dtype=tf.float32)
Update = tf.train.GradientDescentOptimizer(LearnRate).minimize(EntropyLoss+Regularizer)

# 2.5. Classification Diagnostics (how well did we do?).  Note the nice use of
#      `reduce_mean`. 
Prediction = tf.math.round(PredictedProb - Threshold + 0.5)

Recall = (
    tf.reduce_sum(tf.multiply(Prediction, TrueY)) /
    tf.reduce_sum(                        TrueY )  
)

Precision = (
    tf.reduce_sum(tf.multiply(Prediction, TrueY)) /
    tf.reduce_sum(            Prediction        )  
)

###############################################################################
#                                 3. RUN GRAPH                                #
###############################################################################

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # 3.0. Repeatedly... 
    for i in range(NB_UPDATES+1):
        if i%100==0:
            # 3.1. Report the model's accuracy:
            batch_x, batch_t, batch_y = get_batch('test') 
            precision, recall, loss, reg = sess.run([Precision, Recall, EntropyLoss, Regularizer], feed_dict={
                TrueX:batch_x, TrueT:batch_t, TrueY:batch_y, LearnRate:LEARN_RATE, Threshold:0.2 
            }) 
            print('iter %4d \t precision: %.3f \t recall: %.2f \t loss: %.3f \t reg: %.1e' %
                    (i, precision, recall, loss, reg))

        #            ...sample a batch of training data:
        batch_x, batch_t, batch_y = get_batch() 
        #            ...run the gradient descent update on that batch:
        sess.run(Update, feed_dict={
            TrueX:batch_x, TrueT:batch_t, TrueY:batch_y, LearnRate:LEARN_RATE
        }) 
