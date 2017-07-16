
# coding: utf-8

# In[ ]:

#!/disk/scratch/mlp/miniconda2/bin/python


# In[ ]:




# In[1]:
import math
import os
import tensorflow as tf
import numpy as np
import datetime
from mlp.data_providers import CIFAR10DataProvider, CIFAR100DataProvider
import matplotlib.pyplot as plt

# In[2]:

random_seed=142857
rng = np.random.RandomState(random_seed)
train_data = CIFAR10DataProvider('train', batch_size=50, rng=rng)
valid_data = CIFAR10DataProvider('valid', batch_size=50, rng=rng)


# In[3]:

inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
keep_prob_conv=tf.placeholder(tf.float32)
keep_prob_fc=tf.placeholder(tf.float32)


# In[4]:

def weight_variable(name,shape):
    dtype=tf.float32
    kernel_weights = tf.get_variable(name,initializer=tf.truncated_normal(shape, stddev=5e-2))
    return kernel_weights


# In[5]:

def conv_layer(inputs,kernel,strides, output_dim,dropout=False,keep_prob=0.8):
    conv=tf.nn.conv2d(inputs,kernel,strides,padding='SAME')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    pre_activation=tf.nn.bias_add(conv,biases)
    if dropout:
        outputs=tf.nn.dropout(pre_activation,keep_prob)
    else:
        outputs = pre_activation
    return outputs

# In[6]:

def fully_connected_layer(inputs, input_dim, output_dim,dropout=False,keep_prob=0.8):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5),
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    pre_activation=tf.matmul(inputs, weights) + biases
    if dropout:
        outputs=tf.nn.dropout(pre_activation,keep_prob)
    else:
        outputs = pre_activation
    return outputs

# In[8]:

#reshape the flat data to 3-D and add the data augmentation 
n_inputs=tf.reshape(inputs,[50,32,32,3]) # reshape image

# In[9]:
scale1 = tf.Variable(tf.ones([32]))
beta1 = tf.Variable(tf.zeros([32]))

scale = tf.Variable(tf.ones([64]))
beta = tf.Variable(tf.zeros([64]))#offset
epsilon = 1e-3

with tf.name_scope('conv-layer-1'):
    W_conv1=weight_variable('w_conv71',[5,5,3,32])# patch = 5, in size = 3, out size= 64
    conv1 = conv_layer(n_inputs,W_conv1,[1,1,1,1],32,dropout=False, keep_prob=keep_prob_conv)# 32*32*64
    batch_mean1,batch_var1 = tf.nn.moments(conv1,[0,1,2])
    norm1 = tf.nn.batch_normalization(conv1,batch_mean1,batch_var1,beta1,scale1,epsilon)
    activ1 = tf.nn.elu(norm1)
    pool1,row_pooling_sequence1, col_pooling_sequence1 = tf.nn.fractional_max_pool(activ1,pooling_ratio=
                                                                                 [1, math.sqrt(3), math.sqrt(3), 1])
    
with tf.name_scope('conv-layer-2'):
    W_conv2 = weight_variable('w_conv72',[5,5,32,32])# patch=5 in size=64, out size=64
    conv2 = conv_layer(pool1,W_conv2,[1,1,1,1],32,dropout=False,keep_prob=keep_prob_conv)#16*16*64
    batch_mean2,batch_var2 = tf.nn.moments(conv2,[0,1,2])
    norm2 = tf.nn.batch_normalization(conv2,batch_mean2,batch_var2,beta1,scale1,epsilon)
    activ2 = tf.nn.elu(norm2)
    pool2,row_pooling_sequence2, col_pooling_sequence2 = tf.nn.fractional_max_pool(activ2,pooling_ratio=
                                                                                 [1, math.sqrt(3), math.sqrt(3), 1])
    
with tf.name_scope('conv-layer-3'):
    W_conv3 = weight_variable('w_conv73',[5,5,32,64])# patch=5 in size=64, out size=64
    conv3 = conv_layer(pool2,W_conv3,[1,1,1,1],64,dropout=False,keep_prob=keep_prob_conv)#16*16*64
    batch_mean3,batch_var3= tf.nn.moments(conv3,[0,1,2])
    norm3 = tf.nn.batch_normalization(conv3,batch_mean3,batch_var3,beta,scale,epsilon)
    activ3 = tf.nn.elu(norm3)
    pool3,row_pooling_sequence3, col_pooling_sequence3 = tf.nn.fractional_max_pool(activ3,pooling_ratio=
                                                                                 [1, math.sqrt(3), math.sqrt(3), 1])
#with tf.name_scope('conv-layer-4'):
    #W_conv4 = weight_variable('w_conv74',[5,5,64,64])# patch=5 in size=64, out size=64
    #conv4 = conv_layer(pool3,W_conv4,[1,1,1,1],64,dropout=False,keep_prob=keep_prob_conv)#16*16*64
    #batch_mean4,batch_var4= tf.nn.moments(conv4,[0,1,2])
    #norm4 = tf.nn.batch_normalization(conv4,batch_mean4,batch_var4,beta,scale,epsilon)
    #activ4 = tf.nn.elu(norm4)
    #pool4,row_pooling_sequence4, col_pooling_sequence4 = tf.nn.fractional_max_pool(activ4,pooling_ratio=
                                                                                 #[1, math.sqrt(3), math.sqrt(3), 1])

with tf.name_scope('fully-connected-layer-1'):
    reshaped_before_out = tf.reshape(pool3,[50,5*5*64])
    fc1=fully_connected_layer(reshaped_before_out,1600,800,dropout=False,keep_prob=keep_prob_fc)

with tf.name_scope('fully-connected-layer-2'):
    fc2=fully_connected_layer(fc1,800,500,dropout=False,keep_prob=keep_prob_fc)
    
with tf.name_scope('output-layer'):
    outputs = fully_connected_layer(fc2,500, train_data.num_classes,dropout=False,keep_prob=keep_prob_fc)

with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets) )
    
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)),
            tf.float32))
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(error)
    
init = tf.global_variables_initializer()


# In[ ]:

acc_train0=[]
acc_valid0=[]
err_train0=[]
err_valid0=[]
train_data.reset()
valid_data.reset()
with tf.Session() as sess:
    sess.run(init)
    for e in range(100):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy],
                feed_dict={inputs: input_batch, targets: target_batch,keep_prob_conv:0.8,keep_prob_fc:0.8})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        err_train0.append(running_error)
        acc_train0.append(running_accuracy)
        print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'

              .format(e + 1, running_error, running_accuracy))
        if (e + 1) % 1 == 0:
            valid_error = 0.
            valid_accuracy = 0.
            for input_batch, target_batch in valid_data:
                batch_error, batch_acc = sess.run(
                    [error, accuracy],
                    feed_dict={inputs: input_batch, targets: target_batch,keep_prob_conv:1.0,keep_prob_fc:1.0})
                valid_error += batch_error
                valid_accuracy += batch_acc
            valid_error /= valid_data.num_batches
            valid_accuracy /= valid_data.num_batches
            acc_valid0.append(valid_accuracy)
            err_valid0.append(valid_error)
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))


# In[ ]:

#!/disk/scratch/mlp/miniconda2/bin/python


# In[ ]:




# In[1]:
import math
import os
import tensorflow as tf
import numpy as np
import datetime
from mlp.data_providers import CIFAR10DataProvider, CIFAR100DataProvider
import matplotlib.pyplot as plt

# In[2]:

random_seed=142857
rng = np.random.RandomState(random_seed)
train_data = CIFAR10DataProvider('train', batch_size=50, rng=rng)
valid_data = CIFAR10DataProvider('valid', batch_size=50, rng=rng)


# In[3]:

inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
keep_prob_conv=tf.placeholder(tf.float32)
keep_prob_fc=tf.placeholder(tf.float32)


# In[4]:

def weight_variable(name,shape):
    dtype=tf.float32
    kernel_weights = tf.get_variable(name,initializer=tf.truncated_normal(shape, stddev=5e-2))
    return kernel_weights


# In[5]:

def conv_layer(inputs,kernel,strides, output_dim,dropout=False,keep_prob=0.8):
    conv=tf.nn.conv2d(inputs,kernel,strides,padding='SAME')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    pre_activation=tf.nn.bias_add(conv,biases)
    if dropout:
        outputs=tf.nn.dropout(pre_activation,keep_prob)
    else:
        outputs = pre_activation
    return outputs

# In[6]:

def fully_connected_layer(inputs, input_dim, output_dim,dropout=False,keep_prob=0.8):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5),
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    pre_activation=tf.matmul(inputs, weights) + biases
    if dropout:
        outputs=tf.nn.dropout(pre_activation,keep_prob)
    else:
        outputs = pre_activation
    return outputs

# In[8]:

#reshape the flat data to 3-D and add the data augmentation 
n_inputs=tf.reshape(inputs,[50,32,32,3]) # reshape image

# In[9]:
scale1 = tf.Variable(tf.ones([32]))
beta1 = tf.Variable(tf.zeros([32]))

scale = tf.Variable(tf.ones([64]))
beta = tf.Variable(tf.zeros([64]))#offset
epsilon = 1e-3

with tf.name_scope('conv-layer-1'):
    W_conv1=weight_variable('w_conv71',[5,5,3,32])# patch = 5, in size = 3, out size= 64
    conv1 = conv_layer(n_inputs,W_conv1,[1,1,1,1],32,dropout=False, keep_prob=keep_prob_conv)# 32*32*64
    batch_mean1,batch_var1 = tf.nn.moments(conv1,[0,1,2])
    norm1 = tf.nn.batch_normalization(conv1,batch_mean1,batch_var1,beta1,scale1,epsilon)
    activ1 = tf.nn.elu(norm1)
    pool1,row_pooling_sequence1, col_pooling_sequence1 = tf.nn.fractional_max_pool(activ1,pooling_ratio=
                                                                                 [1, math.sqrt(3), math.sqrt(3), 1])
    
with tf.name_scope('conv-layer-2'):
    W_conv2 = weight_variable('w_conv72',[5,5,32,32])# patch=5 in size=64, out size=64
    conv2 = conv_layer(pool1,W_conv2,[1,1,1,1],32,dropout=False,keep_prob=keep_prob_conv)#16*16*64
    batch_mean2,batch_var2 = tf.nn.moments(conv2,[0,1,2])
    norm2 = tf.nn.batch_normalization(conv2,batch_mean2,batch_var2,beta1,scale1,epsilon)
    activ2 = tf.nn.elu(norm2)
    pool2,row_pooling_sequence2, col_pooling_sequence2 = tf.nn.fractional_max_pool(activ2,pooling_ratio=
                                                                                 [1, math.sqrt(3), math.sqrt(3), 1])
    
with tf.name_scope('conv-layer-3'):
    W_conv3 = weight_variable('w_conv73',[5,5,32,64])# patch=5 in size=64, out size=64
    conv3 = conv_layer(pool2,W_conv3,[1,1,1,1],64,dropout=False,keep_prob=keep_prob_conv)#16*16*64
    batch_mean3,batch_var3= tf.nn.moments(conv3,[0,1,2])
    norm3 = tf.nn.batch_normalization(conv3,batch_mean3,batch_var3,beta,scale,epsilon)
    activ3 = tf.nn.elu(norm3)
    pool3,row_pooling_sequence3, col_pooling_sequence3 = tf.nn.fractional_max_pool(activ3,pooling_ratio=
                                                                                 [1, math.sqrt(3), math.sqrt(3), 1])
#with tf.name_scope('conv-layer-4'):
    #W_conv4 = weight_variable('w_conv74',[5,5,64,64])# patch=5 in size=64, out size=64
    #conv4 = conv_layer(pool3,W_conv4,[1,1,1,1],64,dropout=False,keep_prob=keep_prob_conv)#16*16*64
    #batch_mean4,batch_var4= tf.nn.moments(conv4,[0,1,2])
    #norm4 = tf.nn.batch_normalization(conv4,batch_mean4,batch_var4,beta,scale,epsilon)
    #activ4 = tf.nn.elu(norm4)
    #pool4,row_pooling_sequence4, col_pooling_sequence4 = tf.nn.fractional_max_pool(activ4,pooling_ratio=
                                                                                 #[1, math.sqrt(3), math.sqrt(3), 1])

with tf.name_scope('fully-connected-layer-1'):
    reshaped_before_out = tf.reshape(pool3,[50,5*5*64])
    fc1=fully_connected_layer(reshaped_before_out,1600,800,dropout=True,keep_prob=keep_prob_fc)

with tf.name_scope('fully-connected-layer-2'):
    fc2=fully_connected_layer(fc1,800,500,dropout=True,keep_prob=keep_prob_fc)
    
with tf.name_scope('output-layer'):
    outputs = fully_connected_layer(fc2,500, train_data.num_classes,dropout=True,keep_prob=keep_prob_fc)

with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets) )
    
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)),
            tf.float32))
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(error)
    
init = tf.global_variables_initializer()


# In[ ]:

acc_train0=[]
acc_valid0=[]
err_train0=[]
err_valid0=[]
train_data.reset()
valid_data.reset()
with tf.Session() as sess:
    sess.run(init)
    for e in range(100):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy],
                feed_dict={inputs: input_batch, targets: target_batch,keep_prob_conv:0.8,keep_prob_fc:0.8})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        err_train0.append(running_error)
        acc_train0.append(running_accuracy)
        print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'

              .format(e + 1, running_error, running_accuracy))
        if (e + 1) % 1 == 0:
            valid_error = 0.
            valid_accuracy = 0.
            for input_batch, target_batch in valid_data:
                batch_error, batch_acc = sess.run(
                    [error, accuracy],
                    feed_dict={inputs: input_batch, targets: target_batch,keep_prob_conv:1.0,keep_prob_fc:1.0})
                valid_error += batch_error
                valid_accuracy += batch_acc
            valid_error /= valid_data.num_batches
            valid_accuracy /= valid_data.num_batches
            acc_valid0.append(valid_accuracy)
            err_valid0.append(valid_error)
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))


# In[ ]:

#!/disk/scratch/mlp/miniconda2/bin/python


# In[ ]:




# In[1]:
import math
import os
import tensorflow as tf
import numpy as np
import datetime
from mlp.data_providers import CIFAR10DataProvider, CIFAR100DataProvider
import matplotlib.pyplot as plt

# In[2]:

random_seed=142857
rng = np.random.RandomState(random_seed)
train_data = CIFAR10DataProvider('train', batch_size=50, rng=rng)
valid_data = CIFAR10DataProvider('valid', batch_size=50, rng=rng)


# In[3]:

inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
keep_prob_conv=tf.placeholder(tf.float32)
keep_prob_fc=tf.placeholder(tf.float32)


# In[4]:

def weight_variable(name,shape):
    dtype=tf.float32
    kernel_weights = tf.get_variable(name,initializer=tf.truncated_normal(shape, stddev=5e-2))
    return kernel_weights


# In[5]:

def conv_layer(inputs,kernel,strides, output_dim,dropout=False,keep_prob=0.8):
    conv=tf.nn.conv2d(inputs,kernel,strides,padding='SAME')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    pre_activation=tf.nn.bias_add(conv,biases)
    if dropout:
        outputs=tf.nn.dropout(pre_activation,keep_prob)
    else:
        outputs = pre_activation
    return outputs

# In[6]:

def fully_connected_layer(inputs, input_dim, output_dim,dropout=False,keep_prob=0.8):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5),
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    pre_activation=tf.matmul(inputs, weights) + biases
    if dropout:
        outputs=tf.nn.dropout(pre_activation,keep_prob)
    else:
        outputs = pre_activation
    return outputs

# In[8]:

#reshape the flat data to 3-D and add the data augmentation 
n_inputs=tf.reshape(inputs,[50,32,32,3]) # reshape image

# In[9]:
scale1 = tf.Variable(tf.ones([32]))
beta1 = tf.Variable(tf.zeros([32]))

scale = tf.Variable(tf.ones([64]))
beta = tf.Variable(tf.zeros([64]))#offset
epsilon = 1e-3

with tf.name_scope('conv-layer-1'):
    W_conv1=weight_variable('w_conv71',[5,5,3,32])# patch = 5, in size = 3, out size= 64
    conv1 = conv_layer(n_inputs,W_conv1,[1,1,1,1],32,dropout=False, keep_prob=keep_prob_conv)# 32*32*64
    batch_mean1,batch_var1 = tf.nn.moments(conv1,[0,1,2])
    norm1 = tf.nn.batch_normalization(conv1,batch_mean1,batch_var1,beta1,scale1,epsilon)
    activ1 = tf.nn.elu(norm1)
    pool1,row_pooling_sequence1, col_pooling_sequence1 = tf.nn.fractional_max_pool(activ1,pooling_ratio=
                                                      [1, math.sqrt(3), math.sqrt(3), 1],pseudo_random=True)
    
with tf.name_scope('conv-layer-2'):
    W_conv2 = weight_variable('w_conv72',[5,5,32,32])# patch=5 in size=64, out size=64
    conv2 = conv_layer(pool1,W_conv2,[1,1,1,1],32,dropout=False,keep_prob=keep_prob_conv)#16*16*64
    batch_mean2,batch_var2 = tf.nn.moments(conv2,[0,1,2])
    norm2 = tf.nn.batch_normalization(conv2,batch_mean2,batch_var2,beta1,scale1,epsilon)
    activ2 = tf.nn.elu(norm2)
    pool2,row_pooling_sequence2, col_pooling_sequence2 = tf.nn.fractional_max_pool(activ2,pooling_ratio=
                                                                                 [1, math.sqrt(3), math.sqrt(3), 1],pseudo_random=True)
    
with tf.name_scope('conv-layer-3'):
    W_conv3 = weight_variable('w_conv73',[5,5,32,64])# patch=5 in size=64, out size=64
    conv3 = conv_layer(pool2,W_conv3,[1,1,1,1],64,dropout=False,keep_prob=keep_prob_conv)#16*16*64
    batch_mean3,batch_var3= tf.nn.moments(conv3,[0,1,2])
    norm3 = tf.nn.batch_normalization(conv3,batch_mean3,batch_var3,beta,scale,epsilon)
    activ3 = tf.nn.elu(norm3)
    pool3,row_pooling_sequence3, col_pooling_sequence3 = tf.nn.fractional_max_pool(activ3,pooling_ratio=
                                                                                 [1, math.sqrt(3), math.sqrt(3), 1],pseudo_random=True)
#with tf.name_scope('conv-layer-4'):
    #W_conv4 = weight_variable('w_conv74',[5,5,64,64])# patch=5 in size=64, out size=64
    #conv4 = conv_layer(pool3,W_conv4,[1,1,1,1],64,dropout=False,keep_prob=keep_prob_conv)#16*16*64
    #batch_mean4,batch_var4= tf.nn.moments(conv4,[0,1,2])
    #norm4 = tf.nn.batch_normalization(conv4,batch_mean4,batch_var4,beta,scale,epsilon)
    #activ4 = tf.nn.elu(norm4)
    #pool4,row_pooling_sequence4, col_pooling_sequence4 = tf.nn.fractional_max_pool(activ4,pooling_ratio=
                                                                                 #[1, math.sqrt(3), math.sqrt(3), 1])

with tf.name_scope('fully-connected-layer-1'):
    reshaped_before_out = tf.reshape(pool3,[50,5*5*64])
    fc1=fully_connected_layer(reshaped_before_out,1600,800,dropout=False,keep_prob=keep_prob_fc)

with tf.name_scope('fully-connected-layer-2'):
    fc2=fully_connected_layer(fc1,800,500,dropout=False,keep_prob=keep_prob_fc)
    
with tf.name_scope('output-layer'):
    outputs = fully_connected_layer(fc2,500, train_data.num_classes,dropout=False,keep_prob=keep_prob_fc)

with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets) )
    
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)),
            tf.float32))
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(error)
    
init = tf.global_variables_initializer()


# In[ ]:

acc_train0=[]
acc_valid0=[]
err_train0=[]
err_valid0=[]
train_data.reset()
valid_data.reset()
with tf.Session() as sess:
    sess.run(init)
    for e in range(100):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy],
                feed_dict={inputs: input_batch, targets: target_batch,keep_prob_conv:0.8,keep_prob_fc:0.8})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        err_train0.append(running_error)
        acc_train0.append(running_accuracy)
        print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'

              .format(e + 1, running_error, running_accuracy))
        if (e + 1) % 1 == 0:
            valid_error = 0.
            valid_accuracy = 0.
            for input_batch, target_batch in valid_data:
                batch_error, batch_acc = sess.run(
                    [error, accuracy],
                    feed_dict={inputs: input_batch, targets: target_batch,keep_prob_conv:1.0,keep_prob_fc:1.0})
                valid_error += batch_error
                valid_accuracy += batch_acc
            valid_error /= valid_data.num_batches
            valid_accuracy /= valid_data.num_batches
            acc_valid0.append(valid_accuracy)
            err_valid0.append(valid_error)
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))


# In[ ]:

#!/disk/scratch/mlp/miniconda2/bin/python


# In[ ]:




# In[1]:
import math
import os
import tensorflow as tf
import numpy as np
import datetime
from mlp.data_providers import CIFAR10DataProvider, CIFAR100DataProvider
import matplotlib.pyplot as plt

# In[2]:

random_seed=142857
rng = np.random.RandomState(random_seed)
train_data = CIFAR10DataProvider('train', batch_size=50, rng=rng)
valid_data = CIFAR10DataProvider('valid', batch_size=50, rng=rng)


# In[3]:

inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
keep_prob_conv=tf.placeholder(tf.float32)
keep_prob_fc=tf.placeholder(tf.float32)


# In[4]:

def weight_variable(name,shape):
    dtype=tf.float32
    kernel_weights = tf.get_variable(name,initializer=tf.truncated_normal(shape, stddev=5e-2))
    return kernel_weights


# In[5]:

def conv_layer(inputs,kernel,strides, output_dim,dropout=False,keep_prob=0.8):
    conv=tf.nn.conv2d(inputs,kernel,strides,padding='SAME')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    pre_activation=tf.nn.bias_add(conv,biases)
    if dropout:
        outputs=tf.nn.dropout(pre_activation,keep_prob)
    else:
        outputs = pre_activation
    return outputs

# In[6]:

def fully_connected_layer(inputs, input_dim, output_dim,dropout=False,keep_prob=0.8):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5),
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    pre_activation=tf.matmul(inputs, weights) + biases
    if dropout:
        outputs=tf.nn.dropout(pre_activation,keep_prob)
    else:
        outputs = pre_activation
    return outputs

# In[8]:

#reshape the flat data to 3-D and add the data augmentation 
n_inputs=tf.reshape(inputs,[50,32,32,3]) # reshape image

# In[9]:
scale1 = tf.Variable(tf.ones([32]))
beta1 = tf.Variable(tf.zeros([32]))

scale = tf.Variable(tf.ones([64]))
beta = tf.Variable(tf.zeros([64]))#offset
epsilon = 1e-3

with tf.name_scope('conv-layer-1'):
    W_conv1=weight_variable('w_conv71',[5,5,3,32])# patch = 5, in size = 3, out size= 64
    conv1 = conv_layer(n_inputs,W_conv1,[1,1,1,1],32,dropout=False, keep_prob=keep_prob_conv)# 32*32*64
    batch_mean1,batch_var1 = tf.nn.moments(conv1,[0,1,2])
    norm1 = tf.nn.batch_normalization(conv1,batch_mean1,batch_var1,beta1,scale1,epsilon)
    activ1 = tf.nn.elu(norm1)
    pool1,row_pooling_sequence1, col_pooling_sequence1 = tf.nn.fractional_max_pool(activ1,pooling_ratio=
                                                      [1, math.sqrt(3), math.sqrt(3), 1],pseudo_random=True,overlapping=True)
    
with tf.name_scope('conv-layer-2'):
    W_conv2 = weight_variable('w_conv72',[5,5,32,32])# patch=5 in size=64, out size=64
    conv2 = conv_layer(pool1,W_conv2,[1,1,1,1],32,dropout=False,keep_prob=keep_prob_conv)#16*16*64
    batch_mean2,batch_var2 = tf.nn.moments(conv2,[0,1,2])
    norm2 = tf.nn.batch_normalization(conv2,batch_mean2,batch_var2,beta1,scale1,epsilon)
    activ2 = tf.nn.elu(norm2)
    pool2,row_pooling_sequence2, col_pooling_sequence2 = tf.nn.fractional_max_pool(activ2,pooling_ratio=
                                                                                 [1, math.sqrt(3), math.sqrt(3), 1],pseudo_random=True,overlapping=True)
    
with tf.name_scope('conv-layer-3'):
    W_conv3 = weight_variable('w_conv73',[5,5,32,64])# patch=5 in size=64, out size=64
    conv3 = conv_layer(pool2,W_conv3,[1,1,1,1],64,dropout=False,keep_prob=keep_prob_conv)#16*16*64
    batch_mean3,batch_var3= tf.nn.moments(conv3,[0,1,2])
    norm3 = tf.nn.batch_normalization(conv3,batch_mean3,batch_var3,beta,scale,epsilon)
    activ3 = tf.nn.elu(norm3)
    pool3,row_pooling_sequence3, col_pooling_sequence3 = tf.nn.fractional_max_pool(activ3,pooling_ratio=
                                                                                 [1, math.sqrt(3), math.sqrt(3), 1],pseudo_random=True,overlapping=True)
#with tf.name_scope('conv-layer-4'):
    #W_conv4 = weight_variable('w_conv74',[5,5,64,64])# patch=5 in size=64, out size=64
    #conv4 = conv_layer(pool3,W_conv4,[1,1,1,1],64,dropout=False,keep_prob=keep_prob_conv)#16*16*64
    #batch_mean4,batch_var4= tf.nn.moments(conv4,[0,1,2])
    #norm4 = tf.nn.batch_normalization(conv4,batch_mean4,batch_var4,beta,scale,epsilon)
    #activ4 = tf.nn.elu(norm4)
    #pool4,row_pooling_sequence4, col_pooling_sequence4 = tf.nn.fractional_max_pool(activ4,pooling_ratio=
                                                                                 #[1, math.sqrt(3), math.sqrt(3), 1])

with tf.name_scope('fully-connected-layer-1'):
    reshaped_before_out = tf.reshape(pool3,[50,5*5*64])
    fc1=fully_connected_layer(reshaped_before_out,1600,800,dropout=False,keep_prob=keep_prob_fc)

with tf.name_scope('fully-connected-layer-2'):
    fc2=fully_connected_layer(fc1,800,500,dropout=False,keep_prob=keep_prob_fc)
    
with tf.name_scope('output-layer'):
    outputs = fully_connected_layer(fc2,500, train_data.num_classes,dropout=False,keep_prob=keep_prob_fc)

with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets) )
    
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)),
            tf.float32))
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(error)
    
init = tf.global_variables_initializer()


# In[ ]:

acc_train0=[]
acc_valid0=[]
err_train0=[]
err_valid0=[]
train_data.reset()
valid_data.reset()
with tf.Session() as sess:
    sess.run(init)
    for e in range(100):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy],
                feed_dict={inputs: input_batch, targets: target_batch,keep_prob_conv:0.8,keep_prob_fc:0.8})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        err_train0.append(running_error)
        acc_train0.append(running_accuracy)
        print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'

              .format(e + 1, running_error, running_accuracy))
        if (e + 1) % 1 == 0:
            valid_error = 0.
            valid_accuracy = 0.
            for input_batch, target_batch in valid_data:
                batch_error, batch_acc = sess.run(
                    [error, accuracy],
                    feed_dict={inputs: input_batch, targets: target_batch,keep_prob_conv:1.0,keep_prob_fc:1.0})
                valid_error += batch_error
                valid_accuracy += batch_acc
            valid_error /= valid_data.num_batches
            valid_accuracy /= valid_data.num_batches
            acc_valid0.append(valid_accuracy)
            err_valid0.append(valid_error)
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))


# In[ ]:

#!/disk/scratch/mlp/miniconda2/bin/python


# In[ ]:




# In[1]:
import math
import os
import tensorflow as tf
import numpy as np
import datetime
from mlp.data_providers import CIFAR10DataProvider, CIFAR100DataProvider
import matplotlib.pyplot as plt

# In[2]:

random_seed=142857
rng = np.random.RandomState(random_seed)
train_data = CIFAR10DataProvider('train', batch_size=50, rng=rng)
valid_data = CIFAR10DataProvider('valid', batch_size=50, rng=rng)


# In[3]:

inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
keep_prob_conv=tf.placeholder(tf.float32)
keep_prob_fc=tf.placeholder(tf.float32)


# In[4]:

def weight_variable(name,shape):
    dtype=tf.float32
    kernel_weights = tf.get_variable(name,initializer=tf.truncated_normal(shape, stddev=5e-2))
    return kernel_weights


# In[5]:

def conv_layer(inputs,kernel,strides, output_dim,dropout=False,keep_prob=0.8):
    conv=tf.nn.conv2d(inputs,kernel,strides,padding='SAME')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    pre_activation=tf.nn.bias_add(conv,biases)
    if dropout:
        outputs=tf.nn.dropout(pre_activation,keep_prob)
    else:
        outputs = pre_activation
    return outputs

# In[6]:

def fully_connected_layer(inputs, input_dim, output_dim,dropout=False,keep_prob=0.8):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5),
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    pre_activation=tf.matmul(inputs, weights) + biases
    if dropout:
        outputs=tf.nn.dropout(pre_activation,keep_prob)
    else:
        outputs = pre_activation
    return outputs

# In[8]:

#reshape the flat data to 3-D and add the data augmentation 
n_inputs=tf.reshape(inputs,[50,32,32,3]) # reshape image

# In[9]:
scale1 = tf.Variable(tf.ones([32]))
beta1 = tf.Variable(tf.zeros([32]))

scale = tf.Variable(tf.ones([64]))
beta = tf.Variable(tf.zeros([64]))#offset
epsilon = 1e-3

with tf.name_scope('conv-layer-1'):
    W_conv1=weight_variable('w_conv71',[5,5,3,32])# patch = 5, in size = 3, out size= 64
    conv1 = conv_layer(n_inputs,W_conv1,[1,1,1,1],32,dropout=False, keep_prob=keep_prob_conv)# 32*32*64
    batch_mean1,batch_var1 = tf.nn.moments(conv1,[0,1,2])
    norm1 = tf.nn.batch_normalization(conv1,batch_mean1,batch_var1,beta1,scale1,epsilon)
    activ1 = tf.nn.elu(norm1)
    pool1,row_pooling_sequence1, col_pooling_sequence1 = tf.nn.fractional_max_pool(activ1,pooling_ratio=
                                                      [1, math.sqrt(3), math.sqrt(3), 1],overlapping=True)
    
with tf.name_scope('conv-layer-2'):
    W_conv2 = weight_variable('w_conv72',[5,5,32,32])# patch=5 in size=64, out size=64
    conv2 = conv_layer(pool1,W_conv2,[1,1,1,1],32,dropout=False,keep_prob=keep_prob_conv)#16*16*64
    batch_mean2,batch_var2 = tf.nn.moments(conv2,[0,1,2])
    norm2 = tf.nn.batch_normalization(conv2,batch_mean2,batch_var2,beta1,scale1,epsilon)
    activ2 = tf.nn.elu(norm2)
    pool2,row_pooling_sequence2, col_pooling_sequence2 = tf.nn.fractional_max_pool(activ2,pooling_ratio=
                                                                                 [1, math.sqrt(3), math.sqrt(3), 1],overlapping=True)
    
with tf.name_scope('conv-layer-3'):
    W_conv3 = weight_variable('w_conv73',[5,5,32,64])# patch=5 in size=64, out size=64
    conv3 = conv_layer(pool2,W_conv3,[1,1,1,1],64,dropout=False,keep_prob=keep_prob_conv)#16*16*64
    batch_mean3,batch_var3= tf.nn.moments(conv3,[0,1,2])
    norm3 = tf.nn.batch_normalization(conv3,batch_mean3,batch_var3,beta,scale,epsilon)
    activ3 = tf.nn.elu(norm3)
    pool3,row_pooling_sequence3, col_pooling_sequence3 = tf.nn.fractional_max_pool(activ3,pooling_ratio=
                                                                                 [1, math.sqrt(3), math.sqrt(3), 1],overlapping=True)
#with tf.name_scope('conv-layer-4'):
    #W_conv4 = weight_variable('w_conv74',[5,5,64,64])# patch=5 in size=64, out size=64
    #conv4 = conv_layer(pool3,W_conv4,[1,1,1,1],64,dropout=False,keep_prob=keep_prob_conv)#16*16*64
    #batch_mean4,batch_var4= tf.nn.moments(conv4,[0,1,2])
    #norm4 = tf.nn.batch_normalization(conv4,batch_mean4,batch_var4,beta,scale,epsilon)
    #activ4 = tf.nn.elu(norm4)
    #pool4,row_pooling_sequence4, col_pooling_sequence4 = tf.nn.fractional_max_pool(activ4,pooling_ratio=
                                                                                 #[1, math.sqrt(3), math.sqrt(3), 1])

with tf.name_scope('fully-connected-layer-1'):
    reshaped_before_out = tf.reshape(pool3,[50,5*5*64])
    fc1=fully_connected_layer(reshaped_before_out,1600,800,dropout=False,keep_prob=keep_prob_fc)

with tf.name_scope('fully-connected-layer-2'):
    fc2=fully_connected_layer(fc1,800,500,dropout=False,keep_prob=keep_prob_fc)
    
with tf.name_scope('output-layer'):
    outputs = fully_connected_layer(fc2,500, train_data.num_classes,dropout=False,keep_prob=keep_prob_fc)

with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets) )
    
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)),
            tf.float32))
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(error)
    
init = tf.global_variables_initializer()


# In[ ]:

acc_train0=[]
acc_valid0=[]
err_train0=[]
err_valid0=[]
train_data.reset()
valid_data.reset()
with tf.Session() as sess:
    sess.run(init)
    for e in range(100):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy],
                feed_dict={inputs: input_batch, targets: target_batch,keep_prob_conv:0.8,keep_prob_fc:0.8})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        err_train0.append(running_error)
        acc_train0.append(running_accuracy)
        print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'

              .format(e + 1, running_error, running_accuracy))
        if (e + 1) % 1 == 0:
            valid_error = 0.
            valid_accuracy = 0.
            for input_batch, target_batch in valid_data:
                batch_error, batch_acc = sess.run(
                    [error, accuracy],
                    feed_dict={inputs: input_batch, targets: target_batch,keep_prob_conv:1.0,keep_prob_fc:1.0})
                valid_error += batch_error
                valid_accuracy += batch_acc
            valid_error /= valid_data.num_batches
            valid_accuracy /= valid_data.num_batches
            acc_valid0.append(valid_accuracy)
            err_valid0.append(valid_error)
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))


# In[ ]:

#!/disk/scratch/mlp/miniconda2/bin/python


# In[ ]:




# In[1]:

import os
import tensorflow as tf
import numpy as np
import datetime
from mlp.data_providers import CIFAR10DataProvider, CIFAR100DataProvider
import matplotlib.pyplot as plt

# In[2]:

random_seed=142857
rng = np.random.RandomState(random_seed)
train_data = CIFAR10DataProvider('train', batch_size=50, rng=rng)
valid_data = CIFAR10DataProvider('valid', batch_size=50, rng=rng)


# In[3]:

inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
keep_prob_conv=tf.placeholder(tf.float32)
keep_prob_fc=tf.placeholder(tf.float32)


# In[4]:

def weight_variable(name,shape):
    dtype=tf.float32
    kernel_weights = tf.get_variable(name,initializer=tf.truncated_normal(shape, stddev=5e-2))
    return kernel_weights


# In[5]:

def conv_layer(inputs,kernel,strides, output_dim,dropout=False,keep_prob=0.8):
    conv=tf.nn.conv2d(inputs,kernel,strides,padding='SAME')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    pre_activation=tf.nn.bias_add(conv,biases)
    if dropout:
        outputs=tf.nn.dropout(pre_activation,keep_prob)
    else:
        outputs = pre_activation
    return outputs

# In[6]:

def fully_connected_layer(inputs, input_dim, output_dim,dropout=False,keep_prob=0.8):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5),
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    pre_activation=tf.matmul(inputs, weights) + biases
    if dropout:
        outputs=tf.nn.dropout(tf.nn.elu(pre_activation),keep_prob)
    else:
        outputs = tf.nn.elu(pre_activation)
    return outputs

# In[8]:

#reshape the flat data to 3-D and add the data augmentation 
n_inputs=tf.reshape(inputs,[50,32,32,3]) # reshape image

# In[9]:
scale1 = tf.Variable(tf.ones([32]))
beta1 = tf.Variable(tf.zeros([32]))

scale = tf.Variable(tf.ones([64]))
beta = tf.Variable(tf.zeros([64]))#offset
epsilon = 1e-3
with tf.name_scope('conv-layer-1'):
    W_conv1=weight_variable('w_conv71',[5,5,3,32])# patch = 5, in size = 3, out size= 64
    conv1 = conv_layer(n_inputs,W_conv1,[1,1,1,1],32,dropout=True,keep_prob=keep_prob_conv)# 32*32*64
    batch_mean1,batch_var1 = tf.nn.moments(conv1,[0,1,2])
    norm1 = tf.nn.batch_normalization(conv1,batch_mean1,batch_var1,beta1,scale1,epsilon)
    activ1 = tf.nn.elu(norm1)
    pool1 = tf.nn.max_pool(activ1,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME', name='pool1')

with tf.name_scope('conv-layer-2'):
    W_conv2 = weight_variable('w_conv72',[5,5,32,32])# patch=5 in size=64, out size=64
    conv2 = conv_layer(pool1,W_conv2,[1,1,1,1],32,dropout=True,keep_prob=keep_prob_conv)#16*16*64
    batch_mean2,batch_var2 = tf.nn.moments(conv2,[0,1,2])
    norm2 = tf.nn.batch_normalization(conv2,batch_mean2,batch_var2,beta1,scale1,epsilon)
    activ2 = tf.nn.elu(norm2)
    pool2 = tf.nn.max_pool(activ2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME', name='pool2')
    
with tf.name_scope('conv-layer-3'):
    W_conv3 = weight_variable('w_conv73',[5,5,32,64])# patch=5 in size=64, out size=64
    conv3 = conv_layer(pool2,W_conv3,[1,1,1,1],64,dropout=True,keep_prob=keep_prob_conv)#16*16*64
    batch_mean3,batch_var3= tf.nn.moments(conv3,[0,1,2])
    norm3 = tf.nn.batch_normalization(conv3,batch_mean3,batch_var3,beta,scale,epsilon)
    activ3 = tf.nn.elu(norm3)
    pool3 = tf.nn.max_pool(activ3,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME', name='pool3')
                                                                                                                                           

with tf.name_scope('fully-connected-layer-1'):
    reshaped_before_out = tf.reshape(pool3,[50,4*4*64])
    fc1=fully_connected_layer(reshaped_before_out,1024,800,dropout=True,keep_prob=keep_prob_fc)

with tf.name_scope('fully-connected-layer-2'):
    fc2=fully_connected_layer(fc1,800,500,dropout=True,keep_prob=keep_prob_fc)
    
with tf.name_scope('output-layer'):
    outputs = fully_connected_layer(fc2,500, train_data.num_classes,dropout=True,keep_prob=keep_prob_fc)

with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets) )
    
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)),
            tf.float32))
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(error)
    
init = tf.global_variables_initializer()


# In[ ]:

acc_train0=[]
acc_valid0=[]
err_train0=[]
err_valid0=[]
train_data.reset()
valid_data.reset()
with tf.Session() as sess:
    sess.run(init)
    for e in range(100):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy],
                feed_dict={inputs: input_batch, targets: target_batch,keep_prob_conv:0.8,keep_prob_fc:0.8})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        err_train0.append(running_error)
        acc_train0.append(running_accuracy)
        print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'

              .format(e + 1, running_error, running_accuracy))
        if (e + 1) % 1 == 0:
            valid_error = 0.
            valid_accuracy = 0.
            for input_batch, target_batch in valid_data:
                batch_error, batch_acc = sess.run(
                    [error, accuracy],
                    feed_dict={inputs: input_batch, targets: target_batch,keep_prob_conv:1.0,keep_prob_fc:1.0})
                valid_error += batch_error
                valid_accuracy += batch_acc
            valid_error /= valid_data.num_batches
            valid_accuracy /= valid_data.num_batches
            acc_valid0.append(valid_accuracy)
            err_valid0.append(valid_error)
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))

