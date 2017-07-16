
# coding: utf-8

# baseline

# In[1]:

import numpy as np
from mlp.layers import Layer

class TanhLayer(Layer):
    """Layer implementing an element-wise hyperbolic tangent transformation."""

    def fprop(self, inputs):
        """Forward propagates activations through the layer transformation.

        For inputs `x` and outputs `y` this corresponds to `y = tanh(x)`.
        """
        return np.tanh(inputs)

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs.
        """
        return (1. - outputs**2) * grads_wrt_outputs

    def __repr__(self):
        return 'TanhLayer'
    

class ReluLayer(Layer):
    """Layer implementing an element-wise rectified linear transformation."""

    def fprop(self, inputs):
        """Forward propagates activations through the layer transformation.

        For inputs `x` and outputs `y` this corresponds to `y = max(0, x)`.
        """
        return np.maximum(inputs, 0.)

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs.
        """
        return (outputs > 0) * grads_wrt_outputs

    def __repr__(self):
        return 'ReluLayer'


# In[2]:

import numpy as np
from collections import OrderedDict
import logging
from mlp.layers import AffineLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import GlorotUniformInit, ConstantInit
from mlp.learning_rules import MomentumLearningRule
from mlp.data_providers import MNISTDataProvider
from mlp.optimisers import Optimiser
from mlp.data_providers import AugmentedMNISTDataProvider

# Seed a random number generator
seed = 24102016 
rng = np.random.RandomState(seed)

num_epochs = 100
stats_interval = 5
batch_size = 50
learning_rate = 0.01
mom_coeff = 0.9
weights_init_gain = 0.5
biases_init = 0.

# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# Create data provider objects for the MNIST data set
train_data = MNISTDataProvider('train', batch_size, rng=rng)
valid_data = MNISTDataProvider('valid', batch_size, rng=rng)
input_dim, output_dim, hidden_dim = 784, 10, 100

weights_init = GlorotUniformInit(weights_init_gain, rng)
biases_init = ConstantInit(biases_init)
error = CrossEntropySoftmaxError()
learning_rule = MomentumLearningRule(learning_rate, mom_coeff)
data_monitors = {'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}


# In[52]:

weights_penalties = [
    None,
]
run_info = OrderedDict()
models = OrderedDict()
for weights_penalty in weights_penalties:
    # Reset random number generator and data provider states on each run
    # to ensure reproducibility of results
    rng.seed(seed)
    train_data.reset()
    valid_data.reset()
    print('Regularisation: {0}'.format(weights_penalty))
    model = MultipleLayerModel([
        AffineLayer(input_dim, hidden_dim, weights_init, 
                    biases_init, weights_penalty),
        ReluLayer(),
        AffineLayer(hidden_dim, hidden_dim, weights_init, 
                    biases_init, weights_penalty),
        ReluLayer(),
        AffineLayer(hidden_dim, output_dim, weights_init, 
                    biases_init, weights_penalty)
    ])
    optimiser = Optimiser(model, error, learning_rule, train_data, valid_data, data_monitors)
    run_info[weights_penalty] = optimiser.train(num_epochs, stats_interval)
    models[weights_penalty] = model


# In[53]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.style.use('ggplot')

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
for weight_penalty, run in run_info.items():
    stats, keys, run_time = run
    ax1.plot(np.arange(1, stats.shape[0]) * stats_interval, 
             stats[1:, keys['acc(train)']], label=str(weight_penalty))
    ax2.plot(np.arange(1, stats.shape[0]) * stats_interval, 
             stats[1:, keys['acc(valid)']], label=str(weight_penalty))

ax1.legend(loc=0)
ax1.set_xlabel('Epoch number')
ax1.set_ylabel('Training set accuracy')
ax2.legend(loc=0)
ax2.set_xlabel('Epoch number')
ax2.set_ylabel('Validation set accuracy')


# In[12]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.style.use('ggplot')

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
for weight_penalty, run in run_info.items():
    stats, keys, run_time = run
    ax1.plot(np.arange(1, stats.shape[0]) * stats_interval, 
             stats[1:, keys['error(train)']], label=str(weight_penalty))
    ax2.plot(np.arange(1, stats.shape[0]) * stats_interval, 
             stats[1:, keys['error(valid)']], label=str(weight_penalty))
ax1.legend(loc=0)
ax1.set_xlabel('Epoch number')
ax1.set_ylabel('Training set error')
ax1.set_yscale('log')
ax2.legend(loc=0)
ax2.set_xlabel('Epoch number')
ax2.set_ylabel('Validation set error')
ax2.set_yscale('log')


# # definition of zoom and shift

# In[40]:

from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
 
def zooming(inputs,rng):
    orig_ims = inputs.reshape((-1, 28, 28))
    new_ims = orig_ims.copy()
    indices = rng.choice(orig_ims.shape[0], orig_ims.shape[0] // 4, False)
    zoom_rate = rng.uniform(1.0,2.0,size=indices.shape[0])                
    for i, j in enumerate(indices):
        new = zoom(orig_ims[j],zoom_rate[i],order=1)
        width = new.shape[1]
        height = new.shape[0]
        new_ims[j] = new[(height-28)/2:(height+28)/2,(width-28)/2:(width+28)/2]
    return new_ims.reshape((-1, 784)) 


# In[41]:

# passing the test
def show_batch_of_images(img_batch, fig_size=(3, 3)):
    fig = plt.figure(figsize=fig_size)
    batch_size, im_height, im_width = img_batch.shape
    # calculate no. columns per grid row to give square grid
    grid_size = int(batch_size**0.5)
    # intialise empty array to tile image grid into
    tiled = np.empty((im_height * grid_size, 
                      im_width * batch_size // grid_size))
    # iterate over images in batch + indexes within batch
    for i, img in enumerate(img_batch):
        # calculate grid row and column indices
        r, c = i % grid_size, i // grid_size
        tiled[r * im_height:(r + 1) * im_height, 
              c * im_height:(c + 1) * im_height] = img
    ax = fig.add_subplot(111)
    ax.imshow(tiled, cmap='Greys') #, vmin=0., vmax=1.)
    ax.axis('off')
    fig.tight_layout()
    plt.show()
    return fig, ax

test_data = MNISTDataProvider('test', 100, rng=rng)
inputs, targets = test_data.next()
_ = show_batch_of_images(inputs.reshape((-1, 28, 28)))

transformed_inputs = zooming(inputs,rng)
_ = show_batch_of_images(transformed_inputs.reshape((-1, 28, 28)))


# In[34]:

from scipy.ndimage.interpolation import shift
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
 
def shifting(inputs,rng):
    orig_ims = inputs.reshape((-1, 28, 28))
    new_ims = orig_ims.copy()
    indices = rng.choice(orig_ims.shape[0], orig_ims.shape[0] // 4, False)
    shift_rate = rng.uniform(-1.0,1.0,size=indices.shape[0])*8                
    for i, j in enumerate(indices):
        new_ims[j] = shift(orig_ims[j],shift_rate[i],order=1)   
    return new_ims.reshape((-1, 784)) 


# In[35]:

# passing the test
def show_batch_of_images(img_batch, fig_size=(3, 3)):
    fig = plt.figure(figsize=fig_size)
    batch_size, im_height, im_width = img_batch.shape
    # calculate no. columns per grid row to give square grid
    grid_size = int(batch_size**0.5)
    # intialise empty array to tile image grid into
    tiled = np.empty((im_height * grid_size, 
                      im_width * batch_size // grid_size))
    # iterate over images in batch + indexes within batch
    for i, img in enumerate(img_batch):
        # calculate grid row and column indices
        r, c = i % grid_size, i // grid_size
        tiled[r * im_height:(r + 1) * im_height, 
              c * im_height:(c + 1) * im_height] = img
    ax = fig.add_subplot(111)
    ax.imshow(tiled, cmap='Greys') #, vmin=0., vmax=1.)
    ax.axis('off')
    fig.tight_layout()
    plt.show()
    return fig, ax

test_data = MNISTDataProvider('test', 100, rng=rng)
inputs, targets = test_data.next()
_ = show_batch_of_images(inputs.reshape((-1, 28, 28)))

transformed_inputs = shifting(inputs,rng)
_ = show_batch_of_images(transformed_inputs.reshape((-1, 28, 28)))


# # training with zoom

# In[44]:

def zooming_3(inputs, rng):
    orig_ims = inputs.reshape((-1, 28, 28))
    new_ims = orig_ims.copy()
    indices = rng.choice(orig_ims.shape[0], orig_ims.shape[0] // 4, False)
    angles = rng.uniform(1.0,3.0,size=indices.shape[0])                
    for i, j in enumerate(indices):
        new=zoom(orig_ims[j],angles[i],order=1)
        w=new.shape[1]
        h=new.shape[0]
        new_ims[j]=new[(h-28)/2:(h+28)/2,(w-28)/2:(w+28)/2]
    return new_ims.reshape((-1, 784)) 

def zooming_4(inputs, rng):
    orig_ims = inputs.reshape((-1, 28, 28))
    new_ims = orig_ims.copy()
    indices = rng.choice(orig_ims.shape[0], orig_ims.shape[0] // 4, False)
    angles = rng.uniform(1.0, 4.0,size=indices.shape[0])                
    for i, j in enumerate(indices):
        new=zoom(orig_ims[j],angles[i],order=1)
        w=new.shape[1]
        h=new.shape[0]
        new_ims[j]=new[(h-28)/2:(h+28)/2,(w-28)/2:(w+28)/2]
    return new_ims.reshape((-1, 784)) 


# In[46]:

batch_size = 100
num_epochs = 100
learning_rate = 0.01
mom_coeff = 0.9
stats_interval = 5
rng = np.random.RandomState(seed)
run_info = OrderedDict()
models = OrderedDict()

zoom_train_data = AugmentedMNISTDataProvider('train', rng=rng, transformer=zooming)
zoom_train_data3 = AugmentedMNISTDataProvider('train', rng=rng, transformer=zooming_3)
zoom_train_data4 = AugmentedMNISTDataProvider('train', rng=rng, transformer=zooming_4)    
train_datas= [train_data,zoom_train_data,zoom_train_data3,zoom_train_data4]

list=[None,[1,2],[1,3],[1,4]]
i=0

for aug_train_data in train_datas:
    rng.seed(seed)
    aug_train_data.reset()
    valid_data.reset()
    aug_train_data.batch_size = batch_size 
    valid_data.batch_size = batch_size

    weights_init = GlorotUniformInit(0.5, rng=rng)
    biases_init = ConstantInit(0.)
    print('zooming:{0}'.format(list[i]))
    model = MultipleLayerModel([
        AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
        ReluLayer(),
        AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
        ReluLayer(),
        AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
    ])
    i=i+1
    error = CrossEntropySoftmaxError()
    learning_rule = MomentumLearningRule(learning_rate=learning_rate, mom_coeff=mom_coeff)
    data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}
    optimiser = Optimiser(model, error, learning_rule, aug_train_data, valid_data, data_monitors)
    run_info[aug_train_data] = optimiser.train(num_epochs, stats_interval)
    models[aug_train_data] = model


# In[48]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.style.use('ggplot')

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

i=0
for aug_train_data, run in run_info.items():
    stats, keys, run_time = run
    ax1.plot(np.arange(1, stats.shape[0]) * stats_interval, 
             stats[1:, keys['acc(train)']],label=list[i])
    ax2.plot(np.arange(1, stats.shape[0]) * stats_interval, 
            stats[1:, keys['acc(valid)']], label=list[i])
    i=i+1;

ax1.legend(loc=0)
ax1.set_xlabel('Epoch number')
ax1.set_ylabel('Training set accuracy')
ax2.legend(loc=0)
ax2.set_xlabel('Epoch number')
ax2.set_ylabel('Validation set accuracy')


# different data propotion(20%,25%,33%) with zooming[1,2]

# In[3]:

def zooming3(inputs, rng):
    orig_ims = inputs.reshape((-1, 28, 28))
    new_ims = orig_ims.copy()
    indices = rng.choice(orig_ims.shape[0], orig_ims.shape[0] // 3, False)
    angles = rng.uniform(1.0,2.0,size=indices.shape[0])                
    for i, j in enumerate(indices):
        new=zoom(orig_ims[j],angles[i],order=1)
        w=new.shape[1]
        h=new.shape[0]
        new_ims[j]=new[(h-28)/2:(h+28)/2,(w-28)/2:(w+28)/2]
    return new_ims.reshape((-1, 784)) 

def zooming4(inputs, rng):
    orig_ims = inputs.reshape((-1, 28, 28))
    new_ims = orig_ims.copy()
    indices = rng.choice(orig_ims.shape[0], orig_ims.shape[0] // 4, False)
    angles = rng.uniform(1.0, 2.0,size=indices.shape[0])                
    for i, j in enumerate(indices):
        new=zoom(orig_ims[j],angles[i],order=1)
        w=new.shape[1]
        h=new.shape[0]
        new_ims[j]=new[(h-28)/2:(h+28)/2,(w-28)/2:(w+28)/2]
    return new_ims.reshape((-1, 784)) 

def zooming5(inputs, rng):
    orig_ims = inputs.reshape((-1, 28, 28))
    new_ims = orig_ims.copy()
    indices = rng.choice(orig_ims.shape[0], orig_ims.shape[0] // 5, False)
    angles = rng.uniform(1.0,2.0,size=indices.shape[0])                
    for i, j in enumerate(indices):
        new=zoom(orig_ims[j],angles[i],order=1)
        w=new.shape[1]
        h=new.shape[0]
        new_ims[j]=new[(h-28)/2:(h+28)/2,(w-28)/2:(w+28)/2]
    return new_ims.reshape((-1, 784)) 


# In[5]:

from scipy.ndimage.interpolation import zoom

batch_size = 100
num_epochs = 100
learning_rate = 0.01
mom_coeff = 0.9
stats_interval = 5
rng = np.random.RandomState(seed)
run_info = OrderedDict()
models = OrderedDict()

zooming_train_data3 = AugmentedMNISTDataProvider('train', rng=rng, transformer=zooming3)
zooming_train_data4 = AugmentedMNISTDataProvider('train', rng=rng, transformer=zooming4)
zooming_train_data5 = AugmentedMNISTDataProvider('train', rng=rng, transformer=zooming5)    
train_datas= [train_data,zooming_train_data3,zooming_train_data4,zooming_train_data5]

list=[None,3,4,5]
i=0

for aug_train_data in train_datas:
    rng.seed(seed)
    aug_train_data.reset()
    valid_data.reset()
    aug_train_data.batch_size = batch_size 
    valid_data.batch_size = batch_size

    weights_init = GlorotUniformInit(0.5, rng=rng)
    biases_init = ConstantInit(0.)
    print('data propotion:{0}'.format(list[i]))
    model = MultipleLayerModel([
        AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
        ReluLayer(),
        AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
        ReluLayer(),
        AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
    ])
    i=i+1
    
    error = CrossEntropySoftmaxError()
    learning_rule = MomentumLearningRule(learning_rate=learning_rate, mom_coeff=mom_coeff)
    data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}
    optimiser = Optimiser(model, error, learning_rule, aug_train_data, valid_data, data_monitors)
    run_info[aug_train_data] = optimiser.train(num_epochs, stats_interval)
    models[aug_train_data] = model


# In[6]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.style.use('ggplot')

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

i=0
for aug_train_data, run in run_info.items():
    stats, keys, run_time = run
    ax1.plot(np.arange(1, stats.shape[0]) * stats_interval, 
             stats[1:, keys['acc(train)']],label=list[i])
    ax2.plot(np.arange(1, stats.shape[0]) * stats_interval, 
            stats[1:, keys['acc(valid)']], label=list[i])
    i=i+1;

ax1.legend(loc=0)
ax1.set_xlabel('Epoch number')
ax1.set_ylabel('Training set accuracy')
ax2.legend(loc=0)
ax2.set_xlabel('Epoch number')
ax2.set_ylabel('Validation set accuracy')


# # training with shift

# In[54]:

def shifting_6(inputs,rng):
    orig_ims = inputs.reshape((-1, 28, 28))
    new_ims = orig_ims.copy()
    indices = rng.choice(orig_ims.shape[0], orig_ims.shape[0] // 4, False)
    shift_rate = rng.uniform(-1.0,1.0,size=indices.shape[0])*6                
    for i, j in enumerate(indices):
        new_ims[j] = shift(orig_ims[j],shift_rate[i],order=1)   
    return new_ims.reshape((-1, 784)) 

def shifting_10(inputs,rng):
    orig_ims = inputs.reshape((-1, 28, 28))
    new_ims = orig_ims.copy()
    indices = rng.choice(orig_ims.shape[0], orig_ims.shape[0] // 4, False)
    shift_rate = rng.uniform(-1.0,1.0,size=indices.shape[0])*10                
    for i, j in enumerate(indices):
        new_ims[j] = shift(orig_ims[j],shift_rate[i],order=1)   
    return new_ims.reshape((-1, 784)) 


# In[56]:

batch_size = 100
num_epochs = 100
learning_rate = 0.01
mom_coeff = 0.9
stats_interval = 5
rng = np.random.RandomState(seed)
run_info = OrderedDict()
models = OrderedDict()

shift_train_data6 = AugmentedMNISTDataProvider('train', rng=rng, transformer=shifting_6)
shift_train_data = AugmentedMNISTDataProvider('train', rng=rng, transformer=shifting)
shift_train_data10 = AugmentedMNISTDataProvider('train', rng=rng, transformer=shifting_10)    
train_datas= [train_data,shift_train_data6,shift_train_data,shift_train_data10]

list=[None,6,8,10]
i=0

for aug_train_data in train_datas:
    rng.seed(seed)
    aug_train_data.reset()
    valid_data.reset()
    aug_train_data.batch_size = batch_size 
    valid_data.batch_size = batch_size

    weights_init = GlorotUniformInit(0.5, rng=rng)
    biases_init = ConstantInit(0.)
    print('shifting: {0}'.format(list[i]))
    model = MultipleLayerModel([
        AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
        ReluLayer(),
        AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
        ReluLayer(),
        AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
    ])
    i=i+1
    error = CrossEntropySoftmaxError()
    learning_rule = MomentumLearningRule(learning_rate=learning_rate, mom_coeff=mom_coeff)
    data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}
    optimiser = Optimiser(model, error, learning_rule, aug_train_data, valid_data, data_monitors)
    run_info[aug_train_data] = optimiser.train(num_epochs, stats_interval)
    models[aug_train_data] = model


# In[57]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.style.use('ggplot')

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

i=0
for aug_train_data, run in run_info.items():
    stats, keys, run_time = run
    ax1.plot(np.arange(1, stats.shape[0]) * stats_interval, 
             stats[1:, keys['acc(train)']],label=list[i])
    ax2.plot(np.arange(1, stats.shape[0]) * stats_interval, 
            stats[1:, keys['acc(valid)']], label=list[i])
    i=i+1;

ax1.legend(loc=0)
ax1.set_xlabel('Epoch number')
ax1.set_ylabel('Training set accuracy')
ax2.legend(loc=0)
ax2.set_xlabel('Epoch number')
ax2.set_ylabel('Validation set accuracy')


# different data propotion(20%,25%,33%) with shifting 6

# In[7]:

def shifting3(inputs,rng):
    orig_ims = inputs.reshape((-1, 28, 28))
    new_ims = orig_ims.copy()
    indices = rng.choice(orig_ims.shape[0], orig_ims.shape[0] // 3, False)
    shift_rate = rng.uniform(-1.0,1.0,size=indices.shape[0])*6                
    for i, j in enumerate(indices):
        new_ims[j] = shift(orig_ims[j],shift_rate[i],order=1)   
    return new_ims.reshape((-1, 784)) 

def shifting4(inputs,rng):
    orig_ims = inputs.reshape((-1, 28, 28))
    new_ims = orig_ims.copy()
    indices = rng.choice(orig_ims.shape[0], orig_ims.shape[0] // 4, False)
    shift_rate = rng.uniform(-1.0,1.0,size=indices.shape[0])*6                
    for i, j in enumerate(indices):
        new_ims[j] = shift(orig_ims[j],shift_rate[i],order=1)   
    return new_ims.reshape((-1, 784)) 

def shifting5(inputs,rng):
    orig_ims = inputs.reshape((-1, 28, 28))
    new_ims = orig_ims.copy()
    indices = rng.choice(orig_ims.shape[0], orig_ims.shape[0] // 5, False)
    shift_rate = rng.uniform(-1.0,1.0,size=indices.shape[0])*6                
    for i, j in enumerate(indices):
        new_ims[j] = shift(orig_ims[j],shift_rate[i],order=1)   
    return new_ims.reshape((-1, 784)) 


# In[8]:

from scipy.ndimage.interpolation import shift

batch_size = 100
num_epochs = 100
learning_rate = 0.01
mom_coeff = 0.9
stats_interval = 5
rng = np.random.RandomState(seed)
run_info = OrderedDict()
models = OrderedDict()

shift_train_data3 = AugmentedMNISTDataProvider('train', rng=rng, transformer=shifting3)
shift_train_data4 = AugmentedMNISTDataProvider('train', rng=rng, transformer=shifting4)
shift_train_data5 = AugmentedMNISTDataProvider('train', rng=rng, transformer=shifting5)    
train_datas= [train_data,shift_train_data3,shift_train_data4,shift_train_data5]

list=[None,3,4,5]
i=0

for aug_train_data in train_datas:
    rng.seed(seed)
    aug_train_data.reset()
    valid_data.reset()
    aug_train_data.batch_size = batch_size 
    valid_data.batch_size = batch_size

    weights_init = GlorotUniformInit(0.5, rng=rng)
    biases_init = ConstantInit(0.)
    print('shifting 6 data propotion: {0}'.format(list[i]))
    model = MultipleLayerModel([
        AffineLayer(input_dim, hidden_dim, weights_init, biases_init),
        ReluLayer(),
        AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
        ReluLayer(),
        AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
    ])
    i=i+1
    error = CrossEntropySoftmaxError()
    learning_rule = MomentumLearningRule(learning_rate=learning_rate, mom_coeff=mom_coeff)
    data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}
    optimiser = Optimiser(model, error, learning_rule, aug_train_data, valid_data, data_monitors)
    run_info[aug_train_data] = optimiser.train(num_epochs, stats_interval)
    models[aug_train_data] = model


# In[9]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.style.use('ggplot')

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

i=0
for aug_train_data, run in run_info.items():
    stats, keys, run_time = run
    ax1.plot(np.arange(1, stats.shape[0]) * stats_interval, 
             stats[1:, keys['acc(train)']],label=list[i])
    ax2.plot(np.arange(1, stats.shape[0]) * stats_interval, 
            stats[1:, keys['acc(valid)']], label=list[i])
    i=i+1;

ax1.legend(loc=0)
ax1.set_xlabel('Epoch number')
ax1.set_ylabel('Training set accuracy')
ax2.legend(loc=0)
ax2.set_xlabel('Epoch number')
ax2.set_ylabel('Validation set accuracy')

