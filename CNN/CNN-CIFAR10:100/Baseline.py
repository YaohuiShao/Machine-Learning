
# coding: utf-8

# # CIFAR－10

# In[1]:

import os
import tensorflow as tf
import numpy as np
from mlp.data_providers import CIFAR10DataProvider, CIFAR100DataProvider
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[2]:

#random_seed=142784
#rng = np.random.RandomState(random_seed)
#train_data = CIFAR10DataProvider('train', batch_size=50,rng=rng)
#valid_data = CIFAR10DataProvider('valid', batch_size=50,rng=rng)
train_data = CIFAR10DataProvider('train', batch_size=50)
valid_data = CIFAR10DataProvider('valid', batch_size=50)


# # Activation Function Comparision

# In[3]:

from collections import OrderedDict
train_err = OrderedDict()
train_acc = OrderedDict()
valid_err = OrderedDict()
valid_acc = OrderedDict()


# ReLu

# In[190]:

def fully_connected_layer(inputs, input_dim, output_dim, nonlinearity=tf.nn.relu):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    return outputs

inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
num_hidden = 300

list1 = []
list2 = []
list3 = []
list4 = []

with tf.name_scope('fc-layer-1'):
    hidden_1 = fully_connected_layer(inputs, train_data.inputs.shape[1], num_hidden)
with tf.name_scope('fc-layer-2'):
    hidden_2 = fully_connected_layer(hidden_1, num_hidden, num_hidden)
with tf.name_scope('output-layer'):
    outputs = fully_connected_layer(hidden_2,num_hidden,train_data.num_classes,tf.identity)
    
with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

with tf.name_scope('train'):
    #train_step = tf.train.AdamOptimizer().minimize(error)
     train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01, use_locking=False, name='GradientDescent').minimize(error)
init = tf.global_variables_initializer()

#train_data.reset()
#valid_data.reset()

with tf.Session() as sess:
    sess.run(init)
    for e in range(50):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        
        list1.append(running_error)
        list2.append(running_accuracy)
        train_err[111] = (list1)
        train_acc[111] = (list2)
        
        valid_error = 0.
        valid_accuracy = 0.
        for input_batch, target_batch in valid_data:
            batch_error, batch_acc = sess.run(
                [error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            valid_error += batch_error
            valid_accuracy += batch_acc
        valid_error /= valid_data.num_batches
        valid_accuracy /= valid_data.num_batches
            
        list3.append(valid_error)
        list4.append(valid_accuracy)
        valid_err[111] = (list3)
        valid_acc[111] = (list4)
        
        if (e + 1) % 5 == 0:         
            print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
              .format(e + 1, running_error, running_accuracy))
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))


# elu

# In[188]:

def fully_connected_layer4(inputs, input_dim, output_dim, nonlinearity=tf.nn.elu):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    return outputs
inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
num_hidden = 300

list1 = []
list2 = []
list3 = []
list4 = []

with tf.name_scope('fc-layer-1'):
    hidden_1 = fully_connected_layer4(inputs, train_data.inputs.shape[1], num_hidden)
with tf.name_scope('fc-layer-2'):
    hidden_2 = fully_connected_layer4(hidden_1, num_hidden, num_hidden)

with tf.name_scope('output-layer'):
    outputs = fully_connected_layer4(hidden_2,num_hidden,train_data.num_classes,tf.identity)
with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

with tf.name_scope('train'):
    #train_step = tf.train.AdamOptimizer().minimize(error)
     train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01, use_locking=False, name='GradientDescent').minimize(error)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for e in range(50):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        
        list1.append(running_error)
        list2.append(running_accuracy)
        train_err[8] = (list1)
        train_acc[8] = (list2)
        
        valid_error = 0.
        valid_accuracy = 0.
        for input_batch, target_batch in valid_data:
            batch_error, batch_acc = sess.run(
                [error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            valid_error += batch_error
            valid_accuracy += batch_acc
        valid_error /= valid_data.num_batches
        valid_accuracy /= valid_data.num_batches
            
        list3.append(valid_error)
        list4.append(valid_accuracy)
        valid_err[8] = (list3)
        valid_acc[8] = (list4)
        
        if (e + 1) % 5 == 0:         
            print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
              .format(e + 1, running_error, running_accuracy))
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))


# sigmoid

# In[183]:

def fully_connected_layer5(inputs, input_dim, output_dim, nonlinearity=tf.sigmoid):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    return outputs
inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
num_hidden = 300

list1 = []
list2 = []
list3 = []
list4 = []

with tf.name_scope('fc-layer-1'):
    hidden_1 = fully_connected_layer5(inputs, train_data.inputs.shape[1], num_hidden)
with tf.name_scope('fc-layer-2'):
    hidden_2 = fully_connected_layer5(hidden_1, num_hidden, num_hidden)

with tf.name_scope('output-layer'):
    outputs = fully_connected_layer5(hidden_2,num_hidden,train_data.num_classes,tf.identity)
with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

with tf.name_scope('train'):
    #train_step = tf.train.AdamOptimizer().minimize(error)
     train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01, use_locking=False, name='GradientDescent').minimize(error)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for e in range(50):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        
        list1.append(running_error)
        list2.append(running_accuracy)
        train_err[9] = (list1)
        train_acc[9] = (list2)
        
        valid_error = 0.
        valid_accuracy = 0.
        for input_batch, target_batch in valid_data:
            batch_error, batch_acc = sess.run(
                [error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            valid_error += batch_error
            valid_accuracy += batch_acc
        valid_error /= valid_data.num_batches
        valid_accuracy /= valid_data.num_batches
            
        list3.append(valid_error)
        list4.append(valid_accuracy)
        valid_err[9] = (list3)
        valid_acc[9] = (list4)
        
        if (e + 1) % 5 == 0:         
            print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
              .format(e + 1, running_error, running_accuracy))
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))


# In[200]:

plt.style.use('ggplot')

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4) 


for i in [111,8,9]:
    if i == 111:
        string = "relu"
        ax1.plot(train_err[i],label = string)
        ax2.plot(valid_err[i],label = string)
        ax3.plot(train_acc[i],label = string)
        ax4.plot(valid_acc[i],label = string)
    elif i==8:
        string = "elu"
        ax1.plot(train_err[i],label = string)
        ax2.plot(valid_err[i],label = string)
        ax3.plot(train_acc[i],label = string)
        ax4.plot(valid_acc[i],label = string)
    elif i==9:
        string = "sigmoid"
        ax1.plot(train_err[i],label = string)
        ax2.plot(valid_err[i],label = string)
        ax3.plot(train_acc[i],label = string)
        ax4.plot(valid_acc[i],label = string)
        
ax1.legend(loc=0)
ax1.set_xlabel('Epoch number')
ax1.set_ylabel('Training error')
ax2.legend(loc=0)
ax2.set_xlabel('Epoch number')
ax2.set_ylabel('Validation error')
#ax2.set_ylim(1.35,)
ax3.legend(loc=10)
ax3.set_xlabel('Epoch number')
ax3.set_ylabel('Training accuracy')
ax4.legend(loc=10)
ax4.set_xlabel('Epoch number')
ax4.set_ylabel('Validation accuracy')
#ax4.set_ylim(0.3,)

fig.savefig('acti_function.pdf')


# elu 5 500

# In[201]:

inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
num_hidden = 500

list1 = []
list2 = []
list3 = []
list4 = []

with tf.name_scope('fc-layer-1'):
    hidden_1 = fully_connected_layer4(inputs, train_data.inputs.shape[1], num_hidden)
with tf.name_scope('fc-layer-2'):
    hidden_2 = fully_connected_layer4(hidden_1,num_hidden,num_hidden)
with tf.name_scope('fc-layer-3'):
    hidden_3 = fully_connected_layer4(hidden_2,num_hidden,num_hidden)
with tf.name_scope('fc-layer-4'):
    hidden_4 = fully_connected_layer4(hidden_3,num_hidden,num_hidden)
with tf.name_scope('fc-layer-5'):
    hidden_5 = fully_connected_layer4(hidden_4,num_hidden,num_hidden)

with tf.name_scope('output-layer'):
    outputs = fully_connected_layer4(hidden_5,num_hidden,train_data.num_classes,tf.identity)
with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

with tf.name_scope('train'):
    #train_step = tf.train.AdamOptimizer().minimize(error)
     train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01, use_locking=False, name='GradientDescent').minimize(error)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for e in range(50):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        
        list1.append(running_error)
        list2.append(running_accuracy)
        train_err[99] = (list1)
        train_acc[99] = (list2)
        
        valid_error = 0.
        valid_accuracy = 0.
        for input_batch, target_batch in valid_data:
            batch_error, batch_acc = sess.run(
                [error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            valid_error += batch_error
            valid_accuracy += batch_acc
        valid_error /= valid_data.num_batches
        valid_accuracy /= valid_data.num_batches
            
        list3.append(valid_error)
        list4.append(valid_accuracy)
        valid_err[99] = (list3)
        valid_acc[99] = (list4)
        
        if (e + 1) % 5 == 0:         
            print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
              .format(e + 1, running_error, running_accuracy))
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))


# In[131]:

inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
num_hidden = 500

list1 = []
list2 = []
list3 = []
list4 = []

with tf.name_scope('fc-layer-1'):
    hidden_1 = fully_connected_layer6(inputs, train_data.inputs.shape[1], num_hidden)
with tf.name_scope('fc-layer-2'):
    hidden_2 = fully_connected_layer6(hidden_1,num_hidden,num_hidden)
with tf.name_scope('fc-layer-3'):
    hidden_3 = fully_connected_layer6(hidden_2,num_hidden,num_hidden)
with tf.name_scope('fc-layer-4'):
    hidden_4 = fully_connected_layer6(hidden_3,num_hidden,num_hidden)
with tf.name_scope('fc-layer-5'):
    hidden_5 = fully_connected_layer6(hidden_4,num_hidden,num_hidden)

with tf.name_scope('output-layer'):
    outputs = fully_connected_layer6(hidden_5,num_hidden,train_data.num_classes,tf.identity)
    
with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

with tf.name_scope('train'):
    #train_step = tf.train.AdamOptimizer().minimize(error)
     train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01, use_locking=False, name='GradientDescent').minimize(error)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for e in range(50):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        
        list1.append(running_error)
        list2.append(running_accuracy)
        train_err[120] = (list1)
        train_acc[120] = (list2)
        
        valid_error = 0.
        valid_accuracy = 0.
        for input_batch, target_batch in valid_data:
            batch_error, batch_acc = sess.run(
                [error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            valid_error += batch_error
            valid_accuracy += batch_acc
        valid_error /= valid_data.num_batches
        valid_accuracy /= valid_data.num_batches
            
        list3.append(valid_error)
        list4.append(valid_accuracy)
        valid_err[120] = (list3)
        valid_acc[120] = (list4)
        
        if (e + 1) % 5 == 0:         
            print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
              .format(e + 1, running_error, running_accuracy))
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))


# In[202]:

plt.style.use('ggplot')

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4) 


for i in [99,120]:
    if i == 99:
        string = "elu"
        ax1.plot(train_err[i],label = string)
        ax2.plot(valid_err[i],label = string)
        ax3.plot(train_acc[i],label = string)
        ax4.plot(valid_acc[i],label = string)
    elif i==120:
        string = "relu"
        ax1.plot(train_err[i],label = string)
        ax2.plot(valid_err[i],label = string)
        ax3.plot(train_acc[i],label = string)
        ax4.plot(valid_acc[i],label = string)
        
ax1.legend(loc=0)
ax1.set_xlabel('Epoch number')
ax1.set_ylabel('Training error')
ax2.legend(loc=0)
ax2.set_xlabel('Epoch number')
ax2.set_ylabel('Validation error')
#ax2.set_ylim(1.35,)
ax3.legend(loc=10)
ax3.set_xlabel('Epoch number')
ax3.set_ylabel('Training accuracy')
ax4.legend(loc=10)
ax4.set_xlabel('Epoch number')
ax4.set_ylabel('Validation accuracy')
#ax4.set_ylim(0.3,)

fig.savefig('elu-relu-500-5.pdf')


# relu 2 1000

# In[209]:

inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
num_hidden = 1000

list1 = []
list2 = []
list3 = []
list4 = []

with tf.name_scope('fc-layer-1'):
    hidden_1 = fully_connected_layer(inputs, train_data.inputs.shape[1],num_hidden)
with tf.name_scope('fc-layer-2'):
    hidden_2 = fully_connected_layer(hidden_1,num_hidden,num_hidden)


with tf.name_scope('output-layer'):
    outputs = fully_connected_layer(hidden_2,num_hidden,train_data.num_classes,tf.identity)
    
with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

with tf.name_scope('train'):
    #train_step = tf.train.AdamOptimizer().minimize(error)
     train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01, use_locking=False, name='GradientDescent').minimize(error)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for e in range(50):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        
        list1.append(running_error)
        list2.append(running_accuracy)
        train_err[135] = (list1)
        train_acc[135] = (list2)
        
        valid_error = 0.
        valid_accuracy = 0.
        for input_batch, target_batch in valid_data:
            batch_error, batch_acc = sess.run(
                [error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            valid_error += batch_error
            valid_accuracy += batch_acc
        valid_error /= valid_data.num_batches
        valid_accuracy /= valid_data.num_batches
            
        list3.append(valid_error)
        list4.append(valid_accuracy)
        valid_err[135] = (list3)
        valid_acc[135] = (list4)
        
        if (e + 1) % 5 == 0:         
            print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
              .format(e + 1, running_error, running_accuracy))
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))


# elu 2 1000

# In[208]:

inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
num_hidden = 1000

list1 = []
list2 = []
list3 = []
list4 = []

with tf.name_scope('fc-layer-1'):
    hidden_1 = fully_connected_layer4(inputs, train_data.inputs.shape[1],num_hidden)
with tf.name_scope('fc-layer-2'):
    hidden_2 = fully_connected_layer4(hidden_1,num_hidden,num_hidden)


with tf.name_scope('output-layer'):
    outputs = fully_connected_layer4(hidden_2,num_hidden,train_data.num_classes,tf.identity)
    
with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

with tf.name_scope('train'):
    #train_step = tf.train.AdamOptimizer().minimize(error)
     train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01, use_locking=False, name='GradientDescent').minimize(error)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for e in range(50):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        
        list1.append(running_error)
        list2.append(running_accuracy)
        train_err[130] = (list1)
        train_acc[130] = (list2)
        
        valid_error = 0.
        valid_accuracy = 0.
        for input_batch, target_batch in valid_data:
            batch_error, batch_acc = sess.run(
                [error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            valid_error += batch_error
            valid_accuracy += batch_acc
        valid_error /= valid_data.num_batches
        valid_accuracy /= valid_data.num_batches
            
        list3.append(valid_error)
        list4.append(valid_accuracy)
        valid_err[130] = (list3)
        valid_acc[130] = (list4)
        
        if (e + 1) % 5 == 0:         
            print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
              .format(e + 1, running_error, running_accuracy))
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))


# In[210]:

plt.style.use('ggplot')

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4) 


for i in [130,135]:
    if i == 130:
        string = "elu"
        ax1.plot(train_err[i],label = string)
        ax2.plot(valid_err[i],label = string)
        ax3.plot(train_acc[i],label = string)
        ax4.plot(valid_acc[i],label = string)
    elif i==135:
        string = "relu"
        ax1.plot(train_err[i],label = string)
        ax2.plot(valid_err[i],label = string)
        ax3.plot(train_acc[i],label = string)
        ax4.plot(valid_acc[i],label = string)
        
ax1.legend(loc=0)
ax1.set_xlabel('Epoch number')
ax1.set_ylabel('Training error')
ax2.legend(loc=0)
ax2.set_xlabel('Epoch number')
ax2.set_ylabel('Validation error')
#ax2.set_ylim(1.35,)
ax3.legend(loc=10)
ax3.set_xlabel('Epoch number')
ax3.set_ylabel('Training accuracy')
ax4.legend(loc=10)
ax4.set_xlabel('Epoch number')
ax4.set_ylabel('Validation accuracy')
#ax4.set_ylim(0.3,)

fig.savefig('elu-relu-1000-2.pdf')


# In[186]:

train_data100 = CIFAR100DataProvider('train', batch_size=50)
valid_data100 = CIFAR100DataProvider('valid', batch_size=50)


# relu cifar 100

# In[187]:


inputs = tf.placeholder(tf.float32, [None, train_data100.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data100.num_classes], 'targets')
num_hidden = 300

list1 = []
list2 = []
list3 = []
list4 = []

with tf.name_scope('fc-layer-1'):
    hidden_1 = fully_connected_layer(inputs, train_data100.inputs.shape[1], num_hidden)
with tf.name_scope('fc-layer-2'):
    hidden_2 = fully_connected_layer(hidden_1, num_hidden, num_hidden)
with tf.name_scope('output-layer'):
    outputs = fully_connected_layer(hidden_2,num_hidden,train_data100.num_classes,tf.identity)
    
with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

with tf.name_scope('train'):
    #train_step = tf.train.AdamOptimizer().minimize(error)
     train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01, use_locking=False, name='GradientDescent').minimize(error)
init = tf.global_variables_initializer()

#train_data.reset()
#valid_data.reset()

with tf.Session() as sess:
    sess.run(init)
    for e in range(50):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data100:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data100.num_batches
        running_accuracy /= train_data100.num_batches
        
        list1.append(running_error)
        list2.append(running_accuracy)
        train_err[1] = (list1)
        train_acc[1] = (list2)
        
        valid_error = 0.
        valid_accuracy = 0.
        for input_batch, target_batch in valid_data100:
            batch_error, batch_acc = sess.run(
                [error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            valid_error += batch_error
            valid_accuracy += batch_acc
        valid_error /= valid_data100.num_batches
        valid_accuracy /= valid_data100.num_batches
            
        list3.append(valid_error)
        list4.append(valid_accuracy)
        valid_err[1] = (list3)
        valid_acc[1] = (list4)
        
        if (e + 1) % 5 == 0:         
            print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
              .format(e + 1, running_error, running_accuracy))
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))


# elu cifar 100

# In[194]:


def fully_connected_layer4(inputs, input_dim, output_dim, nonlinearity=tf.nn.elu):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    return outputs

inputs = tf.placeholder(tf.float32, [None, train_data100.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data100.num_classes], 'targets')
num_hidden = 300

list1 = []
list2 = []
list3 = []
list4 = []

with tf.name_scope('fc-layer-1'):
    hidden_1 = fully_connected_layer4(inputs, train_data100.inputs.shape[1], num_hidden)
with tf.name_scope('fc-layer-2'):
    hidden_2 = fully_connected_layer4(hidden_1, num_hidden, num_hidden)
with tf.name_scope('output-layer'):
    outputs = fully_connected_layer4(hidden_2,num_hidden,train_data100.num_classes,tf.identity)
    
with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

with tf.name_scope('train'):
    #train_step = tf.train.AdamOptimizer().minimize(error)
     train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01, use_locking=False, name='GradientDescent').minimize(error)
init = tf.global_variables_initializer()

#train_data.reset()
#valid_data.reset()

with tf.Session() as sess:
    sess.run(init)
    for e in range(50):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data100:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data100.num_batches
        running_accuracy /= train_data100.num_batches
        
        list1.append(running_error)
        list2.append(running_accuracy)
        train_err[101] = (list1)
        train_acc[101] = (list2)
        
        valid_error = 0.
        valid_accuracy = 0.
        for input_batch, target_batch in valid_data100:
            batch_error, batch_acc = sess.run(
                [error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            valid_error += batch_error
            valid_accuracy += batch_acc
        valid_error /= valid_data100.num_batches
        valid_accuracy /= valid_data100.num_batches
            
        list3.append(valid_error)
        list4.append(valid_accuracy)
        valid_err[101] = (list3)
        valid_acc[101] = (list4)
        
        if (e + 1) % 5 == 0:         
            print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
              .format(e + 1, running_error, running_accuracy))
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))


# In[239]:

plt.style.use('ggplot')

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4) 


for i in [1,101]:
    if i == 1:
        string = "relu"
        ax1.plot(train_err[i],label = string)
        ax2.plot(valid_err[i],label = string)
        ax3.plot(train_acc[i],label = string)
        ax4.plot(valid_acc[i],label = string)
    elif i==101:
        string = "elu"
        ax1.plot(train_err[i],label = string)
        ax2.plot(valid_err[i],label = string)
        ax3.plot(train_acc[i],label = string)
        ax4.plot(valid_acc[i],label = string)
        
ax1.legend(loc=0)
ax1.set_xlabel('Epoch number')
ax1.set_ylabel('Training error')
ax2.legend(loc=0)
ax2.set_xlabel('Epoch number')
ax2.set_ylabel('Validation error')
#ax2.set_ylim(1.35,)
ax3.legend(loc=10)
ax3.set_xlabel('Epoch number')
ax3.set_ylabel('Training accuracy')
ax4.legend(loc=10)
ax4.set_xlabel('Epoch number')
ax4.set_ylabel('Validation accuracy')
#ax4.set_ylim(0.3,)

fig.savefig('relu_vs_elu_cifar100.pdf')


# # Hidden Layer depths and widths

# # Depths 

# 1 hidden layer (500)

# In[144]:

inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
num_hidden = 500

list1 = []
list2 = []
list3 = []
list4 = []

with tf.name_scope('fc-layer-1'):
    hidden_1 = fully_connected_layer6(inputs, train_data.inputs.shape[1], num_hidden)

with tf.name_scope('output-layer'):
    outputs = fully_connected_layer6(hidden_1,num_hidden,train_data.num_classes,tf.identity)
    
with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

with tf.name_scope('train'):
    #train_step = tf.train.AdamOptimizer().minimize(error)
     train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01, use_locking=False, name='GradientDescent').minimize(error)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for e in range(50):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        
        list1.append(running_error)
        list2.append(running_accuracy)
        train_err[20] = (list1)
        train_acc[20] = (list2)
        
        valid_error = 0.
        valid_accuracy = 0.
        for input_batch, target_batch in valid_data:
            batch_error, batch_acc = sess.run(
                [error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            valid_error += batch_error
            valid_accuracy += batch_acc
        valid_error /= valid_data.num_batches
        valid_accuracy /= valid_data.num_batches
            
        list3.append(valid_error)
        list4.append(valid_accuracy)
        valid_err[20] = (list3)
        valid_acc[20] = (list4)
        
        if (e + 1) % 5 == 0:         
            print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
              .format(e + 1, running_error, running_accuracy))
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))


# 2 hidden layers (500 500)

# In[229]:

inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
num_hidden = 500

list1 = []
list2 = []
list3 = []
list4 = []

with tf.name_scope('fc-layer-1'):
    hidden_1 = fully_connected_layer4(inputs, train_data.inputs.shape[1], num_hidden)
with tf.name_scope('fc-layer-2'):
    hidden_2 = fully_connected_layer4(hidden_1,num_hidden,num_hidden)

with tf.name_scope('output-layer'):
    outputs = fully_connected_layer4(hidden_2,num_hidden,train_data.num_classes,tf.identity)
    
with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

with tf.name_scope('train'):
    #train_step = tf.train.AdamOptimizer().minimize(error)
     train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01, use_locking=False, name='GradientDescent').minimize(error)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for e in range(50):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        
        list1.append(running_error)
        list2.append(running_accuracy)
        train_err[121] = (list1)
        train_acc[121] = (list2)
        
        valid_error = 0.
        valid_accuracy = 0.
        for input_batch, target_batch in valid_data:
            batch_error, batch_acc = sess.run(
                [error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            valid_error += batch_error
            valid_accuracy += batch_acc
        valid_error /= valid_data.num_batches
        valid_accuracy /= valid_data.num_batches
            
        list3.append(valid_error)
        list4.append(valid_accuracy)
        valid_err[121] = (list3)
        valid_acc[121] = (list4)
        
        if (e + 1) % 5 == 0:         
            print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
              .format(e + 1, running_error, running_accuracy))
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))


# 5 hidden layers

# In[131]:

inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
num_hidden = 500

list1 = []
list2 = []
list3 = []
list4 = []

with tf.name_scope('fc-layer-1'):
    hidden_1 = fully_connected_layer6(inputs, train_data.inputs.shape[1], num_hidden)
with tf.name_scope('fc-layer-2'):
    hidden_2 = fully_connected_layer6(hidden_1,num_hidden,num_hidden)
with tf.name_scope('fc-layer-3'):
    hidden_3 = fully_connected_layer6(hidden_2,num_hidden,num_hidden)
with tf.name_scope('fc-layer-4'):
    hidden_4 = fully_connected_layer6(hidden_3,num_hidden,num_hidden)
with tf.name_scope('fc-layer-5'):
    hidden_5 = fully_connected_layer6(hidden_4,num_hidden,num_hidden)

with tf.name_scope('output-layer'):
    outputs = fully_connected_layer6(hidden_5,num_hidden,train_data.num_classes,tf.identity)
    
with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

with tf.name_scope('train'):
    #train_step = tf.train.AdamOptimizer().minimize(error)
     train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01, use_locking=False, name='GradientDescent').minimize(error)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for e in range(50):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        
        list1.append(running_error)
        list2.append(running_accuracy)
        train_err[120] = (list1)
        train_acc[120] = (list2)
        
        valid_error = 0.
        valid_accuracy = 0.
        for input_batch, target_batch in valid_data:
            batch_error, batch_acc = sess.run(
                [error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            valid_error += batch_error
            valid_accuracy += batch_acc
        valid_error /= valid_data.num_batches
        valid_accuracy /= valid_data.num_batches
            
        list3.append(valid_error)
        list4.append(valid_accuracy)
        valid_err[120] = (list3)
        valid_acc[120] = (list4)
        
        if (e + 1) % 5 == 0:         
            print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
              .format(e + 1, running_error, running_accuracy))
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))


# In[240]:

plt.style.use('ggplot')

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4) 

for i in [20,121,83]:
    if i == 20:
        string = "1 hidden layer"
        ax1.plot(train_err[i],label = string)
        ax2.plot(valid_err[i],label = string)
        ax3.plot(train_acc[i],label = string)
        ax4.plot(valid_acc[i],label = string)
    elif i == 121:
        string = "2 hidden layers"
        ax1.plot(train_err[i],label = string)
        ax2.plot(valid_err[i],label = string)
        ax3.plot(train_acc[i],label = string)
        ax4.plot(valid_acc[i],label = string)
    elif i == 83:
        string = "3 hidden layers"
        ax1.plot(train_err[i],label = string)
        ax2.plot(valid_err[i],label = string)
        ax3.plot(train_acc[i],label = string)
        ax4.plot(valid_acc[i],label = string)
    
ax1.legend(loc=0)
ax1.set_xlabel('Epoch number')
ax1.set_ylabel('Training error')
ax2.legend(loc=0)
ax2.set_xlabel('Epoch number')
ax2.set_ylabel('Validation error')
ax3.legend(loc=10)
ax3.set_xlabel('Epoch number')
ax3.set_ylabel('Training accuracy')
ax4.legend(loc=10)
ax4.set_xlabel('Epoch number')
ax4.set_ylabel('Validation accuracy')
ax4.set_ylim(0.35,)

fig.savefig('depths.pdf')


# # Widths

# 2 hidden 1000

# In[208]:

inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
num_hidden = 1000

list1 = []
list2 = []
list3 = []
list4 = []

with tf.name_scope('fc-layer-1'):
    hidden_1 = fully_connected_layer4(inputs, train_data.inputs.shape[1],num_hidden)
with tf.name_scope('fc-layer-2'):
    hidden_2 = fully_connected_layer4(hidden_1,num_hidden,num_hidden)


with tf.name_scope('output-layer'):
    outputs = fully_connected_layer4(hidden_2,num_hidden,train_data.num_classes,tf.identity)
    
with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

with tf.name_scope('train'):
    #train_step = tf.train.AdamOptimizer().minimize(error)
     train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01, use_locking=False, name='GradientDescent').minimize(error)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for e in range(50):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        
        list1.append(running_error)
        list2.append(running_accuracy)
        train_err[130] = (list1)
        train_acc[130] = (list2)
        
        valid_error = 0.
        valid_accuracy = 0.
        for input_batch, target_batch in valid_data:
            batch_error, batch_acc = sess.run(
                [error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            valid_error += batch_error
            valid_accuracy += batch_acc
        valid_error /= valid_data.num_batches
        valid_accuracy /= valid_data.num_batches
            
        list3.append(valid_error)
        list4.append(valid_accuracy)
        valid_err[130] = (list3)
        valid_acc[130] = (list4)
        
        if (e + 1) % 5 == 0:         
            print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
              .format(e + 1, running_error, running_accuracy))
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))


# 2 2000

# In[219]:

inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
num_hidden = 2000

list1 = []
list2 = []
list3 = []
list4 = []

with tf.name_scope('fc-layer-1'):
    hidden_1 = fully_connected_layer4(inputs, train_data.inputs.shape[1],num_hidden)
with tf.name_scope('fc-layer-2'):
    hidden_2 = fully_connected_layer4(hidden_1,num_hidden,num_hidden)


with tf.name_scope('output-layer'):
    outputs = fully_connected_layer4(hidden_2,num_hidden,train_data.num_classes,tf.identity)
    
with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

with tf.name_scope('train'):
    #train_step = tf.train.AdamOptimizer().minimize(error)
     train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01, use_locking=False, name='GradientDescent').minimize(error)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for e in range(50):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        
        list1.append(running_error)
        list2.append(running_accuracy)
        train_err[140] = (list1)
        train_acc[140] = (list2)
        
        valid_error = 0.
        valid_accuracy = 0.
        for input_batch, target_batch in valid_data:
            batch_error, batch_acc = sess.run(
                [error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            valid_error += batch_error
            valid_accuracy += batch_acc
        valid_error /= valid_data.num_batches
        valid_accuracy /= valid_data.num_batches
            
        list3.append(valid_error)
        list4.append(valid_accuracy)
        valid_err[140] = (list3)
        valid_acc[140] = (list4)
        
        if (e + 1) % 5 == 0:         
            print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
              .format(e + 1, running_error, running_accuracy))
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))


# In[224]:

plt.style.use('ggplot')

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4) 

for i in [121,8]:
    if i == 121:
        string = "[num_hidden:500]"
        ax1.plot(train_err[i],label = string)
        ax2.plot(valid_err[i],label = string)
        ax3.plot(train_acc[i],label = string)
        ax4.plot(valid_acc[i],label = string)
    elif i == 8:
        string = "[num_hidden:300]"
        ax1.plot(train_err[i],label = string)
        ax2.plot(valid_err[i],label = string)
        ax3.plot(train_acc[i],label = string)
        ax4.plot(valid_acc[i],label = string)
    
ax1.legend(loc=0)
ax1.set_xlabel('Epoch number')
ax1.set_ylabel('Training error')
ax2.legend(loc=0)
ax2.set_xlabel('Epoch number')
ax2.set_ylabel('Validation error')
ax3.legend(loc=10)
ax3.set_xlabel('Epoch number')
ax3.set_ylabel('Training accuracy')
ax4.legend(loc=10)
ax4.set_xlabel('Epoch number')
ax4.set_ylabel('Validation accuracy')

fig.savefig('widths.pdf')


# 5 hidden

# In[123]:

inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
num_hidden = [2000,1000,800,500,200]

list1 = []
list2 = []
list3 = []
list4 = []

with tf.name_scope('fc-layer-1'):
    hidden_1 = fully_connected_layer6(inputs, train_data.inputs.shape[1], num_hidden[0])
with tf.name_scope('fc-layer-2'):
    hidden_2 = fully_connected_layer6(hidden_1,num_hidden[0],num_hidden[1])
with tf.name_scope('fc-layer-3'):
    hidden_3 = fully_connected_layer6(hidden_2,num_hidden[1],num_hidden[2])
with tf.name_scope('fc-layer-4'):
    hidden_4 = fully_connected_layer6(hidden_3,num_hidden[2],num_hidden[3])
with tf.name_scope('fc-layer-5'):
    hidden_5 = fully_connected_layer6(hidden_4,num_hidden[3],num_hidden[4])

with tf.name_scope('output-layer'):
    outputs = fully_connected_layer6(hidden_5,num_hidden[4],train_data.num_classes,tf.identity)
    
with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

with tf.name_scope('train'):
    #train_step = tf.train.AdamOptimizer().minimize(error)
     train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01, use_locking=False, name='GradientDescent').minimize(error)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for e in range(50):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        
        list1.append(running_error)
        list2.append(running_accuracy)
        train_err[12] = (list1)
        train_acc[12] = (list2)
        
        valid_error = 0.
        valid_accuracy = 0.
        for input_batch, target_batch in valid_data:
            batch_error, batch_acc = sess.run(
                [error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            valid_error += batch_error
            valid_accuracy += batch_acc
        valid_error /= valid_data.num_batches
        valid_accuracy /= valid_data.num_batches
            
        list3.append(valid_error)
        list4.append(valid_accuracy)
        valid_err[12] = (list3)
        valid_acc[12] = (list4)
        
        if (e + 1) % 5 == 0:         
            print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
              .format(e + 1, running_error, running_accuracy))
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))


# In[157]:

plt.style.use('ggplot')

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4) 

for i in [120,12]:
    if i == 120:
        string = "[500,500,500,500,500]"
        ax1.plot(train_err[i],label = string)
        ax2.plot(valid_err[i],label = string)
        ax3.plot(train_acc[i],label = string)
        ax4.plot(valid_acc[i],label = string)
    else:
        string = "[2000,1000,800,500,200]"
        ax1.plot(train_err[i],label = string)
        ax2.plot(valid_err[i],label = string)
        ax3.plot(train_acc[i],label = string)
        ax4.plot(valid_acc[i],label = string)
    
ax1.legend(loc=0)
ax1.set_xlabel('Epoch number')
ax1.set_ylabel('Training error')
ax2.legend(loc=0)
ax2.set_xlabel('Epoch number')
ax2.set_ylabel('Validation error')
ax3.legend(loc=10)
ax3.set_xlabel('Epoch number')
ax3.set_ylabel('Training accuracy')
ax4.legend(loc=10)
ax4.set_xlabel('Epoch number')
ax4.set_ylabel('Validation accuracy')


# 2 1000 VS 3 500

# In[228]:

plt.style.use('ggplot')

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4) 

for i in [130,83]:
    if i == 130:
        string = "2 1000"
        ax1.plot(train_err[i],label = string)
        ax2.plot(valid_err[i],label = string)
        ax3.plot(train_acc[i],label = string)
        ax4.plot(valid_acc[i],label = string)
    else:
        string = "3 500"
        ax1.plot(train_err[i],label = string)
        ax2.plot(valid_err[i],label = string)
        ax3.plot(train_acc[i],label = string)
        ax4.plot(valid_acc[i],label = string)
    
ax1.legend(loc=0)
ax1.set_xlabel('Epoch number')
ax1.set_ylabel('Training error')
ax2.legend(loc=0)
ax2.set_xlabel('Epoch number')
ax2.set_ylabel('Validation error')
ax3.legend(loc=10)
ax3.set_xlabel('Epoch number')
ax3.set_ylabel('Training accuracy')
ax4.legend(loc=10)
ax4.set_xlabel('Epoch number')
ax4.set_ylabel('Validation accuracy')


# In[ ]:

plt.style.use('ggplot')

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4) 

for i in [120,12]:
    if i == 120:
        string = "[500,500,500,500,500]"
        ax1.plot(train_err[i],label = string)
        ax2.plot(valid_err[i],label = string)
        ax3.plot(train_acc[i],label = string)
        ax4.plot(valid_acc[i],label = string)
    else:
        string = "[2000,1000,800,500,200]"
        ax1.plot(train_err[i],label = string)
        ax2.plot(valid_err[i],label = string)
        ax3.plot(train_acc[i],label = string)
        ax4.plot(valid_acc[i],label = string)
    
ax1.legend(loc=0)
ax1.set_xlabel('Epoch number')
ax1.set_ylabel('Training error')
ax2.legend(loc=0)
ax2.set_xlabel('Epoch number')
ax2.set_ylabel('Validation error')
ax3.legend(loc=10)
ax3.set_xlabel('Epoch number')
ax3.set_ylabel('Training accuracy')
ax4.legend(loc=10)
ax4.set_xlabel('Epoch number')
ax4.set_ylabel('Validation accuracy')


# # Normalisation

# In[ ]:


inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
num_hidden = 500

hidden_weights = tf.Variable(
    tf.truncated_normal([image_size * image_size, num_hidden_neurons]))
  hidden_biases = tf.Variable(tf.zeros([num_hidden_neurons]))

list1 = []
list2 = []
list3 = []
list4 = []

with tf.name_scope('fc-layer-1'):
    hidden_1 = fully_connected_layer6(inputs, train_data.inputs.shape[1], num_hidden)
with tf.name_scope('fc-layer-2'):
    hidden_2 = fully_connected_layer6(hidden_1,num_hidden,num_hidden)
with tf.name_scope('fc-layer-3'):
    hidden_3 = fully_connected_layer6(hidden_2,num_hidden,num_hidden)
with tf.name_scope('fc-layer-4'):
    hidden_4 = fully_connected_layer6(hidden_3,num_hidden,num_hidden)
with tf.name_scope('fc-layer-5'):
    hidden_5 = fully_connected_layer6(hidden_4,num_hidden,num_hidden)

with tf.name_scope('output-layer'):
    outputs = fully_connected_layer6(hidden_5,num_hidden,train_data.num_classes,tf.identity)
    
with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

with tf.name_scope('train'):
    #train_step = tf.train.AdamOptimizer().minimize(error)
     train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01, use_locking=False, name='GradientDescent').minimize(error)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for e in range(50):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        
        list1.append(running_error)
        list2.append(running_accuracy)
        train_err[12] = (list1)
        train_acc[12] = (list2)
        
        valid_error = 0.
        valid_accuracy = 0.
        for input_batch, target_batch in valid_data:
            batch_error, batch_acc = sess.run(
                [error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            valid_error += batch_error
            valid_accuracy += batch_acc
        valid_error /= valid_data.num_batches
        valid_accuracy /= valid_data.num_batches
            
        list3.append(valid_error)
        list4.append(valid_accuracy)
        valid_err[12] = (list3)
        valid_acc[12] = (list4)
        
        if (e + 1) % 5 == 0:         
            print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
              .format(e + 1, running_error, running_accuracy))
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))


# # regularisation

# In[214]:

def fully_connected_layer_L2(inputs, input_dim, output_dim, nonlinearity=tf.nn.elu):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    return outputs,weights
inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
num_hidden = 500
beta = 0.0001

list1 = []
list2 = []
list3 = []
list4 = []

with tf.name_scope('fc-layer-1'):
    hidden_1,weights_1 = fully_connected_layer_L2(inputs, train_data.inputs.shape[1], num_hidden)
with tf.name_scope('fc-layer-2'):
    hidden_2,weights_2 = fully_connected_layer_L2(hidden_1,num_hidden,num_hidden)
with tf.name_scope('fc-layer-3'):
    hidden_3,weights_3 = fully_connected_layer_L2(hidden_2,num_hidden,num_hidden)
with tf.name_scope('fc-layer-4'):
    hidden_4,weights_4 = fully_connected_layer_L2(hidden_3,num_hidden,num_hidden)
with tf.name_scope('fc-layer-5'):
    hidden_5,weights_5 = fully_connected_layer_L2(hidden_4,num_hidden,num_hidden)

with tf.name_scope('output-layer'):
    outputs,weights_6 = fully_connected_layer_L2(hidden_5,num_hidden,train_data.num_classes,tf.identity)

with tf.name_scope('L2_regularisation'):
    l2_reg = tf.nn.l2_loss(weights_1)+tf.nn.l2_loss(weights_2)+tf.nn.l2_loss(weights_3)+tf.nn.l2_loss(weights_4)+tf.nn.l2_loss(weights_5)+tf.nn.l2_loss(weights_6)
    
    
with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets)) + beta*l2_reg
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

with tf.name_scope('train'):
    #train_step = tf.train.AdamOptimizer().minimize(error)
     train_step = tf.train.GradientDescentOptimizer(learning_rate=0.001, use_locking=False, name='GradientDescent').minimize(error)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for e in range(50):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        
        list1.append(running_error)
        list2.append(running_accuracy)
        train_err[128] = (list1)
        train_acc[128] = (list2)
        
        valid_error = 0.
        valid_accuracy = 0.
        for input_batch, target_batch in valid_data:
            batch_error, batch_acc = sess.run(
                [error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            valid_error += batch_error
            valid_accuracy += batch_acc
        valid_error /= valid_data.num_batches
        valid_accuracy /= valid_data.num_batches
            
        list3.append(valid_error)
        list4.append(valid_accuracy)
        valid_err[128] = (list3)
        valid_acc[128] = (list4)
        
        if (e + 1) % 5 == 0:         
            print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
              .format(e + 1, running_error, running_accuracy))
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))


# In[215]:

plt.style.use('ggplot')

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4) 

for i in [120,128]:
    if i == 120:
        string = "without L2 "
        ax1.plot(train_err[i],label = string)
        ax2.plot(valid_err[i],label = string)
        ax3.plot(train_acc[i],label = string)
        ax4.plot(valid_acc[i],label = string)
    else:
        string = "with L2"
        ax1.plot(train_err[i],label = string)
        ax2.plot(valid_err[i],label = string)
        ax3.plot(train_acc[i],label = string)
        ax4.plot(valid_acc[i],label = string)
    
ax1.legend(loc=0)
ax1.set_xlabel('Epoch number')
ax1.set_ylabel('Training error')
ax2.legend(loc=0)
ax2.set_xlabel('Epoch number')
ax2.set_ylabel('Validation error')
ax3.legend(loc=10)
ax3.set_xlabel('Epoch number')
ax3.set_ylabel('Training accuracy')
ax4.legend(loc=10)
ax4.set_xlabel('Epoch number')
ax4.set_ylabel('Validation accuracy')


# 3 500 elu adam

# In[216]:


inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
num_hidden = 500

list1 = []
list2 = []
list3 = []
list4 = []

with tf.name_scope('fc-layer-1'):
    hidden_1 = fully_connected_layer4(inputs, train_data.inputs.shape[1], num_hidden)
with tf.name_scope('fc-layer-2'):
    hidden_2 = fully_connected_layer4(hidden_1, num_hidden, num_hidden)
with tf.name_scope('fc-layer-3'):
    hidden_3 = fully_connected_layer4(hidden_2, num_hidden, num_hidden)

with tf.name_scope('output-layer'):
    outputs = fully_connected_layer4(hidden_3,num_hidden,train_data.num_classes,tf.identity)
with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

with tf.name_scope('train'):
     train_step = tf.train.AdamOptimizer(learning_rate=0.0001,epsilon=0.001).minimize(error)
    #train_step = tf.train.GradientDescentOptimizer(learning_rate=0.001, use_locking=False, name='GradientDescent').minimize(error)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for e in range(50):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        
        list1.append(running_error)
        list2.append(running_accuracy)
        train_err[83] = (list1)
        train_acc[83] = (list2)
        
        valid_error = 0.
        valid_accuracy = 0.
        for input_batch, target_batch in valid_data:
            batch_error, batch_acc = sess.run(
                [error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            valid_error += batch_error
            valid_accuracy += batch_acc
        valid_error /= valid_data.num_batches
        valid_accuracy /= valid_data.num_batches
            
        list3.append(valid_error)
        list4.append(valid_accuracy)
        valid_err[83] = (list3)
        valid_acc[83] = (list4)
        
        if (e + 1) % 5 == 0:         
            print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
              .format(e + 1, running_error, running_accuracy))
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))


# L2 elu adam 500 3

# In[225]:

def fully_connected_layer_L2(inputs, input_dim, output_dim, nonlinearity=tf.nn.elu):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    return outputs,weights

inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
num_hidden = 500
beta = 0.0001

list1 = []
list2 = []
list3 = []
list4 = []

with tf.name_scope('fc-layer-1'):
    hidden_1,weights_1 = fully_connected_layer_L2(inputs, train_data.inputs.shape[1], num_hidden)
with tf.name_scope('fc-layer-2'):
    hidden_2,weights_2 = fully_connected_layer_L2(hidden_1,num_hidden,num_hidden)
with tf.name_scope('fc-layer-3'):
    hidden_3,weights_3 = fully_connected_layer_L2(hidden_2,num_hidden,num_hidden)

with tf.name_scope('output-layer'):
    outputs,weights_4 = fully_connected_layer_L2(hidden_3,num_hidden,train_data.num_classes,tf.identity)

with tf.name_scope('L2_regularisation'):
    l2_reg = tf.nn.l2_loss(weights_1)+tf.nn.l2_loss(weights_2)+tf.nn.l2_loss(weights_3)+tf.nn.l2_loss(weights_4)
    
    
with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets)) + beta*l2_reg
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate=0.0001,epsilon=0.001).minimize(error)
     #train_step = tf.train.GradientDescentOptimizer(learning_rate=0.001, use_locking=False, name='GradientDescent').minimize(error)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for e in range(50):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        
        list1.append(running_error)
        list2.append(running_accuracy)
        train_err[129] = (list1)
        train_acc[129] = (list2)
        
        valid_error = 0.
        valid_accuracy = 0.
        for input_batch, target_batch in valid_data:
            batch_error, batch_acc = sess.run(
                [error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            valid_error += batch_error
            valid_accuracy += batch_acc
        valid_error /= valid_data.num_batches
        valid_accuracy /= valid_data.num_batches
            
        list3.append(valid_error)
        list4.append(valid_accuracy)
        valid_err[129] = (list3)
        valid_acc[129] = (list4)
        
        if (e + 1) % 5 == 0:         
            print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
              .format(e + 1, running_error, running_accuracy))
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))


# In[226]:

plt.style.use('ggplot')

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4) 

for i in [83,129]:
    if i == 83:
        string = "without L2 "
        ax1.plot(train_err[i],label = string)
        ax2.plot(valid_err[i],label = string)
        ax3.plot(train_acc[i],label = string)
        ax4.plot(valid_acc[i],label = string)
    else:
        string = "with L2"
        ax1.plot(train_err[i],label = string)
        ax2.plot(valid_err[i],label = string)
        ax3.plot(train_acc[i],label = string)
        ax4.plot(valid_acc[i],label = string)
    
ax1.legend(loc=0)
ax1.set_xlabel('Epoch number')
ax1.set_ylabel('Training error')
ax2.legend(loc=0)
ax2.set_xlabel('Epoch number')
ax2.set_ylabel('Validation error')
ax3.legend(loc=10)
ax3.set_xlabel('Epoch number')
ax3.set_ylabel('Training accuracy')
ax4.legend(loc=10)
ax4.set_xlabel('Epoch number')
ax4.set_ylabel('Validation accuracy')


# Test

# In[230]:


inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
num_hidden = 200

list1 = []
list2 = []
list3 = []
list4 = []

with tf.name_scope('fc-layer-1'):
    hidden_1 = fully_connected_layer6(inputs, train_data.inputs.shape[1], num_hidden)


with tf.name_scope('output-layer'):
    outputs = fully_connected_layer6(hidden_1,num_hidden,train_data.num_classes,tf.identity)
with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

with tf.name_scope('train'):
    #train_step = tf.train.AdamOptimizer().minimize(error)
     train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01, use_locking=False, name='GradientDescent').minimize(error)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for e in range(50):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        
        list1.append(running_error)
        list2.append(running_accuracy)
        train_err[888] = (list1)
        train_acc[888] = (list2)
        
        valid_error = 0.
        valid_accuracy = 0.
        for input_batch, target_batch in valid_data:
            batch_error, batch_acc = sess.run(
                [error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            valid_error += batch_error
            valid_accuracy += batch_acc
        valid_error /= valid_data.num_batches
        valid_accuracy /= valid_data.num_batches
            
        list3.append(valid_error)
        list4.append(valid_accuracy)
        valid_err[888] = (list3)
        valid_acc[888] = (list4)
        
        if (e + 1) % 5 == 0:         
            print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
              .format(e + 1, running_error, running_accuracy))
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))


# In[231]:


inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
num_hidden = 200

list1 = []
list2 = []
list3 = []
list4 = []

with tf.name_scope('fc-layer-1'):
    hidden_1 = fully_connected_layer6(inputs, train_data.inputs.shape[1], num_hidden)
with tf.name_scope('fc-layer-2'):
    hidden_2 = fully_connected_layer6(hidden_1, num_hidden, num_hidden)

with tf.name_scope('output-layer'):
    outputs = fully_connected_layer6(hidden_2,num_hidden,train_data.num_classes,tf.identity)
with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

with tf.name_scope('train'):
    #train_step = tf.train.AdamOptimizer().minimize(error)
     train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01, use_locking=False, name='GradientDescent').minimize(error)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for e in range(50):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        
        list1.append(running_error)
        list2.append(running_accuracy)
        train_err[808] = (list1)
        train_acc[808] = (list2)
        
        valid_error = 0.
        valid_accuracy = 0.
        for input_batch, target_batch in valid_data:
            batch_error, batch_acc = sess.run(
                [error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            valid_error += batch_error
            valid_accuracy += batch_acc
        valid_error /= valid_data.num_batches
        valid_accuracy /= valid_data.num_batches
            
        list3.append(valid_error)
        list4.append(valid_accuracy)
        valid_err[808] = (list3)
        valid_acc[808] = (list4)
        
        if (e + 1) % 5 == 0:         
            print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
              .format(e + 1, running_error, running_accuracy))
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))


# In[232]:

def fully_connected_layer4(inputs, input_dim, output_dim, nonlinearity=tf.nn.elu):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    return outputs
inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
num_hidden = 300

list1 = []
list2 = []
list3 = []
list4 = []

with tf.name_scope('fc-layer-1'):
    hidden_1 = fully_connected_layer4(inputs, train_data.inputs.shape[1], num_hidden)
with tf.name_scope('fc-layer-2'):
    hidden_2 = fully_connected_layer4(hidden_1, num_hidden, num_hidden)

with tf.name_scope('output-layer'):
    outputs = fully_connected_layer4(hidden_2,num_hidden,train_data.num_classes,tf.identity)
with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

with tf.name_scope('train'):
    #train_step = tf.train.AdamOptimizer().minimize(error)
     train_step = tf.train.GradientDescentOptimizer(learning_rate=0.001, use_locking=False, name='GradientDescent').minimize(error)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for e in range(100):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        
        list1.append(running_error)
        list2.append(running_accuracy)
        train_err[87] = (list1)
        train_acc[87] = (list2)
        
        valid_error = 0.
        valid_accuracy = 0.
        for input_batch, target_batch in valid_data:
            batch_error, batch_acc = sess.run(
                [error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            valid_error += batch_error
            valid_accuracy += batch_acc
        valid_error /= valid_data.num_batches
        valid_accuracy /= valid_data.num_batches
            
        list3.append(valid_error)
        list4.append(valid_accuracy)
        valid_err[87] = (list3)
        valid_acc[87] = (list4)
        
        if (e + 1) % 5 == 0:         
            print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
              .format(e + 1, running_error, running_accuracy))
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))


# In[233]:

def fully_connected_layer4(inputs, input_dim, output_dim, nonlinearity=tf.nn.elu):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    return outputs
inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
num_hidden = 300

list1 = []
list2 = []
list3 = []
list4 = []

with tf.name_scope('fc-layer-1'):
    hidden_1 = fully_connected_layer4(inputs, train_data.inputs.shape[1], num_hidden)
with tf.name_scope('fc-layer-2'):
    hidden_2 = fully_connected_layer4(hidden_1, num_hidden, num_hidden)

with tf.name_scope('output-layer'):
    outputs = fully_connected_layer4(hidden_2,num_hidden,train_data.num_classes,tf.identity)
with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

with tf.name_scope('train'):
    #train_step = tf.train.AdamOptimizer().minimize(error)
     train_step = tf.train.GradientDescentOptimizer(learning_rate=0.0001, use_locking=False, name='GradientDescent').minimize(error)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for e in range(100):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        
        list1.append(running_error)
        list2.append(running_accuracy)
        train_err[87] = (list1)
        train_acc[87] = (list2)
        
        valid_error = 0.
        valid_accuracy = 0.
        for input_batch, target_batch in valid_data:
            batch_error, batch_acc = sess.run(
                [error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            valid_error += batch_error
            valid_accuracy += batch_acc
        valid_error /= valid_data.num_batches
        valid_accuracy /= valid_data.num_batches
            
        list3.append(valid_error)
        list4.append(valid_accuracy)
        valid_err[87] = (list3)
        valid_acc[87] = (list4)
        
        if (e + 1) % 5 == 0:         
            print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
              .format(e + 1, running_error, running_accuracy))
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))


# In[234]:

#4 1000 nol2
inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
num_hidden = 1000

list1 = []
list2 = []
list3 = []
list4 = []

with tf.name_scope('fc-layer-1'):
    hidden_1= fully_connected_layer4(inputs, train_data.inputs.shape[1], num_hidden)
with tf.name_scope('fc-layer-2'):
    hidden_2= fully_connected_layer4(hidden_1,num_hidden,num_hidden)
with tf.name_scope('fc-layer-3'):
    hidden_3= fully_connected_layer4(hidden_2,num_hidden,num_hidden)
with tf.name_scope('fc-layer-4'):
    hidden_4 = fully_connected_layer4(hidden_3,num_hidden,num_hidden)


with tf.name_scope('output-layer'):
    outputs= fully_connected_layer4(hidden_4,num_hidden,train_data.num_classes,tf.identity)


    
    
with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets)) 
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

with tf.name_scope('train'):
    #train_step = tf.train.AdamOptimizer().minimize(error)
     train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01, use_locking=False, name='GradientDescent').minimize(error)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for e in range(50):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        
        list1.append(running_error)
        list2.append(running_accuracy)
        train_err[821] = (list1)
        train_acc[821] = (list2)
        
        valid_error = 0.
        valid_accuracy = 0.
        for input_batch, target_batch in valid_data:
            batch_error, batch_acc = sess.run(
                [error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            valid_error += batch_error
            valid_accuracy += batch_acc
        valid_error /= valid_data.num_batches
        valid_accuracy /= valid_data.num_batches
            
        list3.append(valid_error)
        list4.append(valid_accuracy)
        valid_err[821] = (list3)
        valid_acc[821] = (list4)
        
        if (e + 1) % 5 == 0:         
            print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
              .format(e + 1, running_error, running_accuracy))
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))


# In[237]:

#l2
def fully_connected_layer_L2(inputs, input_dim, output_dim, nonlinearity=tf.nn.elu):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    return outputs,weights
inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
num_hidden = 1000
beta = 0.001

list1 = []
list2 = []
list3 = []
list4 = []
list5_train=[]
list6_valid=[]

with tf.name_scope('fc-layer-1'):
    hidden_1,weights_1 = fully_connected_layer_L2(inputs, train_data.inputs.shape[1], num_hidden)
with tf.name_scope('fc-layer-2'):
    hidden_2,weights_2 = fully_connected_layer_L2(hidden_1,num_hidden,num_hidden)
with tf.name_scope('fc-layer-3'):
    hidden_3,weights_3 = fully_connected_layer_L2(hidden_2,num_hidden,num_hidden)
with tf.name_scope('fc-layer-4'):
    hidden_4,weights_4 = fully_connected_layer_L2(hidden_3,num_hidden,num_hidden)



with tf.name_scope('output-layer'):
    outputs,weights_5 = fully_connected_layer_L2(hidden_4,num_hidden,train_data.num_classes,tf.identity)

with tf.name_scope('L2_regularisation'):
    l2_reg = tf.nn.l2_loss(weights_1)+tf.nn.l2_loss(weights_2)+tf.nn.l2_loss(weights_3)+tf.nn.l2_loss(weights_4)+tf.nn.l2_loss(weights_5)
    
    
with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets)) + beta*l2_reg
with tf.name_scope('error_withoutl2'):
    error_withoutl2 = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

with tf.name_scope('train'):
    #train_step = tf.train.AdamOptimizer().minimize(error)
     train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01, use_locking=False, name='GradientDescent').minimize(error)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for e in range(50):
        running_error = 0.
        running_accuracy = 0.
        tr_total = 0.
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc,error_withoutl2_tr = sess.run(
                [train_step, error, accuracy,error_withoutl2], 
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
            tr_total += error_withoutl2_tr  
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        tr_total/= train_data.num_batches
        
        list1.append(running_error)
        list2.append(running_accuracy)
        list5_train.append(tr_total)
        train_err[829] = (list1)
        train_acc[829] = (list2)
        
        valid_error = 0.
        valid_accuracy = 0.
        valid_total=0.
        for input_batch, target_batch in valid_data:
            batch_error, batch_acc,error_withoutl2_valid = sess.run(
                [error, accuracy,error_withoutl2], 
                feed_dict={inputs: input_batch, targets: target_batch})
            valid_error += batch_error
            valid_accuracy += batch_acc
            valid_total += error_withoutl2_valid
        valid_error /= valid_data.num_batches
        valid_accuracy /= valid_data.num_batches
        valid_total /= valid_data.num_batches
            
        list3.append(valid_error)
        list4.append(valid_accuracy)
        list6_valid.append(valid_total)
        valid_err[-829] = (list3)
        valid_acc[829] = (list4)
        
        if (e + 1) % 5 == 0:         
            print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
              .format(e + 1, running_error, running_accuracy))
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))


# In[238]:

plt.style.use('ggplot')

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4) 

for i in [821,829]:
    if i == 821:
        string = "without L2 "
        ax1.plot(train_err[i],label = string)
        ax2.plot(valid_err[i],label = string)
        ax3.plot(train_acc[i],label = string)
        ax4.plot(valid_acc[i],label = string)
    else:
        string = "with L2"
        ax1.plot(train_err[i],label = string)
        ax2.plot(valid_err[i],label = string)
        ax3.plot(train_acc[i],label = string)
        ax4.plot(valid_acc[i],label = string)
ax1.plot(list5_train,label ="l2 true error")
ax2.plot(list6_valid,label ="l2 true error")

    
ax1.legend(loc=0)
ax1.set_xlabel('Epoch number')
ax1.set_ylabel('Training error')
ax2.legend(loc=0)
ax2.set_xlabel('Epoch number')
ax2.set_ylabel('Validation error')
ax3.legend(loc=10)
ax3.set_xlabel('Epoch number')
ax3.set_ylabel('Training accuracy')
ax4.legend(loc=10)
ax4.set_xlabel('Epoch number')
ax4.set_ylabel('Validation accuracy')


# In[4]:

#l2
train_data100 = CIFAR100DataProvider('train', batch_size=50)
valid_data100 = CIFAR100DataProvider('valid', batch_size=50)

def fully_connected_layer_L2(inputs, input_dim, output_dim, nonlinearity=tf.nn.elu):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    return outputs,weights
inputs = tf.placeholder(tf.float32, [None, train_data100.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data100.num_classes], 'targets')
num_hidden = 800
beta = 0.001

list1 = []
list2 = []
list3 = []
list4 = []
list7_train=[]
list8_valid=[]

with tf.name_scope('fc-layer-1'):
    hidden_1,weights_1 = fully_connected_layer_L2(inputs, train_data100.inputs.shape[1], num_hidden)
with tf.name_scope('fc-layer-2'):
    hidden_2,weights_2 = fully_connected_layer_L2(hidden_1,num_hidden,num_hidden)
with tf.name_scope('fc-layer-3'):
    hidden_3,weights_3 = fully_connected_layer_L2(hidden_2,num_hidden,num_hidden)
with tf.name_scope('fc-layer-4'):
    hidden_4,weights_4 = fully_connected_layer_L2(hidden_3,num_hidden,num_hidden)



with tf.name_scope('output-layer'):
    outputs,weights_5 = fully_connected_layer_L2(hidden_4,num_hidden,train_data100.num_classes,tf.identity)

with tf.name_scope('L2_regularisation'):
    l2_reg = tf.nn.l2_loss(weights_1)+tf.nn.l2_loss(weights_2)+tf.nn.l2_loss(weights_3)+tf.nn.l2_loss(weights_4)+tf.nn.l2_loss(weights_5)
    
    
with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets)) + beta*l2_reg
with tf.name_scope('error_withoutl2'):
    error_withoutl2 = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

with tf.name_scope('train'):
    #train_step = tf.train.AdamOptimizer().minimize(error)
     train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01, use_locking=False, name='GradientDescent').minimize(error)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for e in range(50):
        running_error = 0.
        running_accuracy = 0.
        tr_total = 0.
        for input_batch, target_batch in train_data100:
            _, batch_error, batch_acc,error_withoutl2_tr = sess.run(
                [train_step, error, accuracy,error_withoutl2], 
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
            tr_total += error_withoutl2_tr  
        running_error /= train_data100.num_batches
        running_accuracy /= train_data100.num_batches
        tr_total/= train_data100.num_batches
        
        list1.append(running_error)
        list2.append(running_accuracy)
        list7_train.append(tr_total)
        train_err[899] = (list1)
        train_acc[899] = (list2)
        
        valid_error = 0.
        valid_accuracy = 0.
        valid_total=0.
        for input_batch, target_batch in valid_data100:
            batch_error, batch_acc,error_withoutl2_valid = sess.run(
                [error, accuracy,error_withoutl2], 
                feed_dict={inputs: input_batch, targets: target_batch})
            valid_error += batch_error
            valid_accuracy += batch_acc
            valid_total += error_withoutl2_valid
        valid_error /= valid_data100.num_batches
        valid_accuracy /= valid_data100.num_batches
        valid_total /= valid_data100.num_batches
            
        list3.append(valid_error)
        list4.append(valid_accuracy)
        list8_valid.append(valid_total)
        valid_err[899] = (list3)
        valid_acc[899] = (list4)
        
        if (e + 1) % 5 == 0:         
            print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
              .format(e + 1, running_error, running_accuracy))
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))


# In[5]:

#l2
train_data100 = CIFAR100DataProvider('train', batch_size=50)
valid_data100 = CIFAR100DataProvider('valid', batch_size=50)

def fully_connected_layer_L2(inputs, input_dim, output_dim, nonlinearity=tf.nn.elu):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    return outputs,weights
inputs = tf.placeholder(tf.float32, [None, train_data100.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data100.num_classes], 'targets')
num_hidden = 800
beta = 0.002

list1 = []
list2 = []
list3 = []
list4 = []
list9_train=[]
list10_valid=[]

with tf.name_scope('fc-layer-1'):
    hidden_1,weights_1 = fully_connected_layer_L2(inputs, train_data100.inputs.shape[1], num_hidden)
with tf.name_scope('fc-layer-2'):
    hidden_2,weights_2 = fully_connected_layer_L2(hidden_1,num_hidden,num_hidden)
with tf.name_scope('fc-layer-3'):
    hidden_3,weights_3 = fully_connected_layer_L2(hidden_2,num_hidden,num_hidden)
with tf.name_scope('fc-layer-4'):
    hidden_4,weights_4 = fully_connected_layer_L2(hidden_3,num_hidden,num_hidden)



with tf.name_scope('output-layer'):
    outputs,weights_5 = fully_connected_layer_L2(hidden_4,num_hidden,train_data100.num_classes,tf.identity)

with tf.name_scope('L2_regularisation'):
    l2_reg = tf.nn.l2_loss(weights_1)+tf.nn.l2_loss(weights_2)+tf.nn.l2_loss(weights_3)+tf.nn.l2_loss(weights_4)+tf.nn.l2_loss(weights_5)
    
    
with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets)) + beta*l2_reg
with tf.name_scope('error_withoutl2'):
    error_withoutl2 = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

with tf.name_scope('train'):
    #train_step = tf.train.AdamOptimizer().minimize(error)
     train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01, use_locking=False, name='GradientDescent').minimize(error)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for e in range(50):
        running_error = 0.
        running_accuracy = 0.
        tr_total = 0.
        for input_batch, target_batch in train_data100:
            _, batch_error, batch_acc,error_withoutl2_tr = sess.run(
                [train_step, error, accuracy,error_withoutl2], 
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
            tr_total += error_withoutl2_tr  
        running_error /= train_data100.num_batches
        running_accuracy /= train_data100.num_batches
        tr_total/= train_data100.num_batches
        
        list1.append(running_error)
        list2.append(running_accuracy)
        list9_train.append(tr_total)
        train_err[799] = (list1)
        train_acc[799] = (list2)
        
        valid_error = 0.
        valid_accuracy = 0.
        valid_total=0.
        for input_batch, target_batch in valid_data100:
            batch_error, batch_acc,error_withoutl2_valid = sess.run(
                [error, accuracy,error_withoutl2], 
                feed_dict={inputs: input_batch, targets: target_batch})
            valid_error += batch_error
            valid_accuracy += batch_acc
            valid_total += error_withoutl2_valid
        valid_error /= valid_data100.num_batches
        valid_accuracy /= valid_data100.num_batches
        valid_total /= valid_data100.num_batches
            
        list3.append(valid_error)
        list4.append(valid_accuracy)
        list10_valid.append(valid_total)
        valid_err[799] = (list3)
        valid_acc[799] = (list4)
        
        if (e + 1) % 5 == 0:         
            print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
              .format(e + 1, running_error, running_accuracy))
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))


# In[6]:

#no L2 cifar-100
def fully_connected_layer4(inputs, input_dim, output_dim, nonlinearity=tf.nn.elu):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5), 
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    return outputs
inputs = tf.placeholder(tf.float32, [None, train_data100.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data100.num_classes], 'targets')
num_hidden = 1000

list1 = []
list2 = []
list3 = []
list4 = []

with tf.name_scope('fc-layer-1'):
    hidden_1 = fully_connected_layer4(inputs, train_data100.inputs.shape[1], num_hidden)
with tf.name_scope('fc-layer-2'):
    hidden_2 = fully_connected_layer4(hidden_1, num_hidden, num_hidden)
with tf.name_scope('fc-layer-3'):
    hidden_3 = fully_connected_layer4(hidden_2,num_hidden,num_hidden)
with tf.name_scope('fc-layer-4'):
    hidden_4 = fully_connected_layer4(hidden_3,num_hidden,num_hidden)



with tf.name_scope('output-layer'):
    outputs = fully_connected_layer4(hidden_4,num_hidden,train_data100.num_classes,tf.identity)
    
with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

with tf.name_scope('train'):
    #train_step = tf.train.AdamOptimizer().minimize(error)
     train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01, use_locking=False, name='GradientDescent').minimize(error)
init = tf.global_variables_initializer()

#train_data.reset()
#valid_data.reset()

with tf.Session() as sess:
    sess.run(init)
    for e in range(50):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data100:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data100.num_batches
        running_accuracy /= train_data100.num_batches
        
        list1.append(running_error)
        list2.append(running_accuracy)
        train_err[189] = (list1)
        train_acc[189] = (list2)
        
        valid_error = 0.
        valid_accuracy = 0.
        for input_batch, target_batch in valid_data100:
            batch_error, batch_acc = sess.run(
                [error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            valid_error += batch_error
            valid_accuracy += batch_acc
        valid_error /= valid_data100.num_batches
        valid_accuracy /= valid_data100.num_batches
            
        list3.append(valid_error)
        list4.append(valid_accuracy)
        valid_err[189] = (list3)
        valid_acc[189] = (list4)
        
        if (e + 1) % 5 == 0:         
            print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
              .format(e + 1, running_error, running_accuracy))
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))


# In[10]:

plt.style.use('ggplot')

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4) 

for i in [189,799]:
    if i == 189:
        string = "without L2 "
        ax1.plot(train_err[i],label = string)
        ax2.plot(valid_err[i],label = string)
        ax3.plot(train_acc[i],label = string)
        ax4.plot(valid_acc[i],label = string)
    else:
        string = "with L2"
        ax1.plot(train_err[i],label = string)
        ax2.plot(valid_err[i],label = string)
        ax3.plot(train_acc[i],label = string)
        ax4.plot(valid_acc[i],label = string)
ax1.plot(list9_train,label ="l2 true error")
ax2.plot(list10_valid,label ="l2 true error")

    
ax1.legend(loc=0)
ax1.set_xlabel('Epoch number')
ax1.set_ylabel('Training error')
ax2.legend(loc=0)
ax2.set_xlabel('Epoch number')
ax2.set_ylabel('Validation error')
ax3.legend(loc=10)
ax3.set_xlabel('Epoch number')
ax3.set_ylabel('Training accuracy')
ax4.legend(loc=10)
ax4.set_xlabel('Epoch number')
ax4.set_ylabel('Validation accuracy')

fig.savefig('cifar100.pdf')


# In[9]:

list10_valid


# In[ ]:



