
# coding: utf-8

# In[ ]:

# Machine Learning Practical: Coursework 1

**Release date: Monday 10th October 2016**  
**Due date: 16:00 Thursday 27th October 2016**

Instructions for the coursework are [available as a PDF here](http://www.inf.ed.ac.uk/teaching/courses/mlp/2016/coursework_1.pdf).


# ## Part 1: Learning rate schedules

# In[5]:

# The below code will set up the data providers, random number
# generator and logger objects needed for training runs. As
# loading the data from file take a little while you generally
# will probably not want to reload the data providers on
# every training run. If you wish to reset their state you
# should instead use the .reset() method of the data providers.
import numpy as np
import logging
from mlp.data_providers import MNISTDataProvider
import matplotlib.pyplot as plt
from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import GradientDescentLearningRule
from mlp.schedulers import TimedependentLearningRateScheduler
from mlp.optimisers import Optimiser
from mlp.schedulers import ConstantLearningRateScheduler
get_ipython().magic(u'matplotlib inline')
plt.style.use('ggplot')


# # Compare timedependent & Constant(1,2)

# In[33]:

# Seed a random number generator
seed = 10102016 
rng = np.random.RandomState(seed)

# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# Create data provider objects for the MNIST data set
train_data = MNISTDataProvider('train', batch_size=50, rng=rng)
valid_data = MNISTDataProvider('valid', batch_size=50, rng=rng)

input_dim, output_dim, hidden_dim = 784, 10, 100

#Set training run hyperparameters
batch_size = 50  # number of data points in a batch
init_scale = 0.01  # scale for random parameter initialisation
num_epochs = 100  # number of training epochs to perform
learning_rate = 0.1
r = 50
stats_interval = 5  # epoch interval between recording and printing stats
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)
data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

optimiser = Optimiser(model, error, learning_rule, train_data, valid_data, data_monitors,schedulers)

stats1, keys1, run_time1 = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)

train_data.batch_size = batch_size 
valid_data.batch_size = batch_size

model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

error = CrossEntropySoftmaxError()

learning_rule = GradientDescentLearningRule(learning_rate=learning_rate)

schedulers = [TimedependentLearningRateScheduler(learning_rate,r)]

_ = stats1, keys1, run_time1


# In[34]:

# Seed a random number generator
seed = 10102016 
rng = np.random.RandomState(seed)

# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# Create data provider objects for the MNIST data set
train_data = MNISTDataProvider('train', batch_size=50, rng=rng)
valid_data = MNISTDataProvider('valid', batch_size=50, rng=rng)

input_dim, output_dim, hidden_dim = 784, 10, 100

#Set training run hyperparameters
batch_size = 50  # number of data points in a batch
init_scale = 0.01  # scale for random parameter initialisation
num_epochs = 100  # number of training epochs to perform
learning_rate = 0.1
stats_interval = 5  # epoch interval between recording and printing stats
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)
data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

optimiser = Optimiser(model, error, learning_rule, train_data, valid_data, data_monitors,schedulers)

stats2, keys2, run_time2 = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)

train_data.batch_size = batch_size 
valid_data.batch_size = batch_size

model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

error = CrossEntropySoftmaxError()

learning_rule = GradientDescentLearningRule(learning_rate=learning_rate)

schedulers = [ConstantLearningRateScheduler(learning_rate)]

_ = stats2, keys2, run_time2


# In[35]:

fig_1 = plt.figure(figsize=(8, 4))
ax_1 = fig_1.add_subplot(111)
for k in ['error(train)', 'error(valid)']:
    ax_1.plot(np.arange(1, stats1.shape[0]) * stats_interval, stats1[1:, keys1[k]], label=k+"timedependent")
   
    ax_1.plot(np.arange(1, stats2.shape[0]) * stats_interval, stats2[1:, keys2[k]], label=k+"constant")
    
ax_1.legend(loc=0)
ax_1.set_xlabel('Epoch number')


fig_2 = plt.figure(figsize=(8, 4))
ax_2 = fig_2.add_subplot(111)
for k in ['acc(train)', 'acc(valid)']:
    ax_2.plot(np.arange(1, stats1.shape[0]) * stats_interval, stats1[1:, keys1[k]], label=k+"timedependent")
    ax_2.plot(np.arange(1, stats2.shape[0]) * stats_interval, stats2[1:, keys2[k]], label=k+"constant")

ax_2.legend(loc=0)
ax_2.set_xlabel('Epoch number')
    
_=fig_1, ax_1, fig_2, ax_2
    
fig_1.savefig('1.pdf')
fig_2.savefig('2.pdf')
fig_1.savefig('1.svg')
fig_2.savefig('2.svg')


# # using TimedependentLearningRateScheduler

# change eta0 (3,4)

# In[36]:

# Seed a random number generator
seed = 10102016 
rng = np.random.RandomState(seed)

# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# Create data provider objects for the MNIST data set
train_data = MNISTDataProvider('train', batch_size=50, rng=rng)
valid_data = MNISTDataProvider('valid', batch_size=50, rng=rng)

input_dim, output_dim, hidden_dim = 784, 10, 100
#Set training run hyperparameters
batch_size = 50  # number of data points in a batch
init_scale = 0.1  # scale for random parameter initialisation
num_epochs = 100  # number of training epochs to perform
learning_rate = 0.001
r = 50
stats_interval = 5  # epoch interval between recording and printing stats
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)
data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}
optimiser = Optimiser(model, error, learning_rule, train_data, valid_data, data_monitors,schedulers)

stats1, keys1, run_time1 = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)


model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

error = CrossEntropySoftmaxError()

learning_rule = GradientDescentLearningRule(learning_rate=learning_rate)

schedulers = [TimedependentLearningRateScheduler(learning_rate,r)]

_ = stats1, keys1, run_time1


# In[38]:

# Seed a random number generator
seed = 10102016 
rng = np.random.RandomState(seed)

# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# Create data provider objects for the MNIST data set
train_data = MNISTDataProvider('train', batch_size=50, rng=rng)
valid_data = MNISTDataProvider('valid', batch_size=50, rng=rng)

input_dim, output_dim, hidden_dim = 784, 10, 100

#Set training run hyperparameters
batch_size = 50  # number of data points in a batch
init_scale = 0.01  # scale for random parameter initialisation
num_epochs = 100  # number of training epochs to perform
learning_rate = 0.01
r = 50
stats_interval = 5  # epoch interval between recording and printing stats
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)
data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

optimiser = Optimiser(model, error, learning_rule, train_data, valid_data, data_monitors,schedulers)

stats2, keys2, run_time2 = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)

train_data.batch_size = batch_size 
valid_data.batch_size = batch_size

model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

error = CrossEntropySoftmaxError()

learning_rule = GradientDescentLearningRule(learning_rate=learning_rate)

schedulers = [TimedependentLearningRateScheduler(learning_rate,r)]

_ = stats2, keys2, run_time2


# In[40]:

# Seed a random number generator
seed = 10102016 
rng = np.random.RandomState(seed)

# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# Create data provider objects for the MNIST data set
train_data = MNISTDataProvider('train', batch_size=50, rng=rng)
valid_data = MNISTDataProvider('valid', batch_size=50, rng=rng)

input_dim, output_dim, hidden_dim = 784, 10, 100

#Set training run hyperparameters
batch_size = 50  # number of data points in a batch
init_scale = 0.1  # scale for random parameter initialisation
num_epochs = 100  # number of training epochs to perform
learning_rate = 0.5
r = 50
stats_interval = 5  # epoch interval between recording and printing stats
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)
data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}
optimiser = Optimiser(model, error, learning_rule, train_data, valid_data, data_monitors,schedulers)

stats3, keys3, run_time3 = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)


train_data.batch_size = batch_size 
valid_data.batch_size = batch_size

model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

error = CrossEntropySoftmaxError()

learning_rule = GradientDescentLearningRule(learning_rate=learning_rate)

schedulers = [TimedependentLearningRateScheduler(learning_rate,r)]

_ = stats3, keys3, run_time3


# In[41]:

fig_1 = plt.figure(figsize=(8, 4))
ax_1 = fig_1.add_subplot(111)
for k in ['error(train)', 'error(valid)']:
    ax_1.plot(np.arange(1, stats1.shape[0]) * stats_interval, stats1[1:, keys1[k]], label=k+"learningrate=0.001")
    ax_1.plot(np.arange(1, stats2.shape[0]) * stats_interval, stats2[1:, keys2[k]], label=k+"learningrate=0.01")
    ax_1.plot(np.arange(1, stats3.shape[0]) * stats_interval, stats3[1:, keys3[k]], label=k+"learningrate=0.5")
ax_1.legend(loc=0)
ax_1.set_xlabel('Epoch number')


fig_2 = plt.figure(figsize=(8, 4))
ax_2 = fig_2.add_subplot(111)
for k in ['acc(train)', 'acc(valid)']:
    ax_2.plot(np.arange(1, stats1.shape[0]) * stats_interval, stats1[1:, keys1[k]], label=k+"learningrate=0.001")
    ax_2.plot(np.arange(1, stats2.shape[0]) * stats_interval, stats2[1:, keys2[k]], label=k+"learningrate=0.01")
    ax_2.plot(np.arange(1, stats3.shape[0]) * stats_interval, stats3[1:, keys3[k]], label=k+"learningrate=0.5")
ax_2.legend(loc=0)
ax_2.set_xlabel('Epoch number')
    
_=fig_1, ax_1, fig_2, ax_2
    
fig_1.savefig('3.pdf')
fig_2.savefig('4.pdf')
fig_1.savefig('3.svg')
fig_2.savefig('4.svg')


# Change r (5,6)

# In[48]:

# Seed a random number generator
seed = 10102016 
rng = np.random.RandomState(seed)

# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# Create data provider objects for the MNIST data set
train_data = MNISTDataProvider('train', batch_size=50, rng=rng)
valid_data = MNISTDataProvider('valid', batch_size=50, rng=rng)

input_dim, output_dim, hidden_dim = 784, 10, 100
#Set training run hyperparameters
batch_size = 50  # number of data points in a batch
init_scale = 0.1  # scale for random parameter initialisation
num_epochs = 100  # number of training epochs to perform
learning_rate = 0.01
r = 5
stats_interval = 5  # epoch interval between recording and printing stats
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)
data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}
optimiser = Optimiser(model, error, learning_rule, train_data, valid_data, data_monitors,schedulers)

stats1, keys1, run_time1 = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)


model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

error = CrossEntropySoftmaxError()

learning_rule = GradientDescentLearningRule(learning_rate=learning_rate)

schedulers = [TimedependentLearningRateScheduler(learning_rate,r)]

_ = stats1, keys1, run_time1


# In[44]:

# Seed a random number generator
seed = 10102016 
rng = np.random.RandomState(seed)

# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# Create data provider objects for the MNIST data set
train_data = MNISTDataProvider('train', batch_size=50, rng=rng)
valid_data = MNISTDataProvider('valid', batch_size=50, rng=rng)

input_dim, output_dim, hidden_dim = 784, 10, 100
#Set training run hyperparameters
batch_size = 50  # number of data points in a batch
init_scale = 0.1  # scale for random parameter initialisation
num_epochs = 100  # number of training epochs to perform
learning_rate = 0.01
r = 50
stats_interval = 5  # epoch interval between recording and printing stats
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)
data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}
optimiser = Optimiser(model, error, learning_rule, train_data, valid_data, data_monitors,schedulers)

stats2, keys2, run_time2 = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)


model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

error = CrossEntropySoftmaxError()

learning_rule = GradientDescentLearningRule(learning_rate=learning_rate)

schedulers = [TimedependentLearningRateScheduler(learning_rate,r)]

_ = stats2, keys2, run_time2


# In[49]:

# Seed a random number generator
seed = 10102016 
rng = np.random.RandomState(seed)

# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# Create data provider objects for the MNIST data set
train_data = MNISTDataProvider('train', batch_size=50, rng=rng)
valid_data = MNISTDataProvider('valid', batch_size=50, rng=rng)

input_dim, output_dim, hidden_dim = 784, 10, 100
#Set training run hyperparameters
batch_size = 50  # number of data points in a batch
init_scale = 0.1  # scale for random parameter initialisation
num_epochs = 100  # number of training epochs to perform
learning_rate = 0.01
r = 5000
stats_interval = 5  # epoch interval between recording and printing stats
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)
data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}
optimiser = Optimiser(model, error, learning_rule, train_data, valid_data, data_monitors,schedulers)

stats3, keys3, run_time3 = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)


model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

error = CrossEntropySoftmaxError()

learning_rule = GradientDescentLearningRule(learning_rate=learning_rate)

schedulers = [TimedependentLearningRateScheduler(learning_rate,r)]

_ = stats3, keys3, run_time3


# In[50]:

fig_1 = plt.figure(figsize=(8, 4))
ax_1 = fig_1.add_subplot(111)
for k in ['error(train)', 'error(valid)']:
    ax_1.plot(np.arange(1, stats1.shape[0]) * stats_interval, stats1[1:, keys1[k]], label=k+"r=5")
    ax_1.plot(np.arange(1, stats2.shape[0]) * stats_interval, stats2[1:, keys2[k]], label=k+"r=50")
    ax_1.plot(np.arange(1, stats3.shape[0]) * stats_interval, stats3[1:, keys3[k]], label=k+"r=5000")
ax_1.legend(loc=0)
ax_1.set_xlabel('Epoch number')


fig_2 = plt.figure(figsize=(8, 4))
ax_2 = fig_2.add_subplot(111)
for k in ['acc(train)', 'acc(valid)']:
    ax_2.plot(np.arange(1, stats1.shape[0]) * stats_interval, stats1[1:, keys1[k]], label=k+"r=5")
    ax_2.plot(np.arange(1, stats2.shape[0]) * stats_interval, stats2[1:, keys2[k]], label=k+"r=50")
    ax_2.plot(np.arange(1, stats3.shape[0]) * stats_interval, stats3[1:, keys3[k]], label=k+"r=5000")
ax_2.legend(loc=0)
ax_2.set_xlabel('Epoch number')
    
_=fig_1, ax_1, fig_2, ax_2
    
fig_1.savefig('5.pdf')
fig_2.savefig('6.pdf')
fig_1.savefig('5.svg')
fig_2.savefig('6.svg')


# ## Part 2: Momentum learning rule

# In[51]:

import numpy as np
import logging
from mlp.data_providers import MNISTDataProvider
import matplotlib.pyplot as plt
from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import GradientDescentLearningRule
from mlp.schedulers import TimedependentLearningRateScheduler
from mlp.optimisers import Optimiser
from mlp.schedulers import ConstantLearningRateScheduler
import matplotlib.pyplot as plt
from mlp.learning_rules import MomentumLearningRule

get_ipython().magic(u'matplotlib inline')
plt.style.use('ggplot')


# In[52]:

# Seed a random number generator
seed = 10102016 
rng = np.random.RandomState(seed)

# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# Create data provider objects for the MNIST data set
train_data = MNISTDataProvider('train', batch_size=50, rng=rng)
valid_data = MNISTDataProvider('valid', batch_size=50, rng=rng)

input_dim, output_dim, hidden_dim = 784, 10, 100

#Set training run hyperparameters
batch_size = 50  # number of data points in a batch
init_scale = 0.01  # scale for random parameter initialisation
num_epochs = 100  # number of training epochs to perform
learning_rate = 0.1
mom_coeff = 0.1
stats_interval = 5  # epoch interval between recording and printing stats
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)
data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

optimiser = Optimiser(model, error, learning_rule, train_data, valid_data, data_monitors)

stats1, keys1, run_time1 = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)

train_data.batch_size = batch_size 
valid_data.batch_size = batch_size

model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

error = CrossEntropySoftmaxError()

learning_rule = MomentumLearningRule(learning_rate=learning_rate, mom_coeff=mom_coeff)

_ = stats1, keys1, run_time1


# In[53]:

# Seed a random number generator
seed = 10102016 
rng = np.random.RandomState(seed)

# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# Create data provider objects for the MNIST data set
train_data = MNISTDataProvider('train', batch_size=50, rng=rng)
valid_data = MNISTDataProvider('valid', batch_size=50, rng=rng)

input_dim, output_dim, hidden_dim = 784, 10, 100

#Set training run hyperparameters
batch_size = 50  # number of data points in a batch
init_scale = 0.01  # scale for random parameter initialisation
num_epochs = 100  # number of training epochs to perform
learning_rate = 0.1
mom_coeff = 0.5
stats_interval = 5  # epoch interval between recording and printing stats
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)
data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

optimiser = Optimiser(model, error, learning_rule, train_data, valid_data, data_monitors)

stats2, keys2, run_time2 = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)

train_data.batch_size = batch_size 
valid_data.batch_size = batch_size

model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

error = CrossEntropySoftmaxError()

learning_rule = MomentumLearningRule(learning_rate=learning_rate, mom_coeff=mom_coeff)


_ = stats2, keys2, run_time2


# In[54]:

# Seed a random number generator
seed = 10102016 
rng = np.random.RandomState(seed)

# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# Create data provider objects for the MNIST data set
train_data = MNISTDataProvider('train', batch_size=50, rng=rng)
valid_data = MNISTDataProvider('valid', batch_size=50, rng=rng)

input_dim, output_dim, hidden_dim = 784, 10, 100

#Set training run hyperparameters
batch_size = 50  # number of data points in a batch
init_scale = 0.01  # scale for random parameter initialisation
num_epochs = 100  # number of training epochs to perform
learning_rate = 0.1
mom_coeff = 0.9
stats_interval = 5  # epoch interval between recording and printing stats
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)
data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

optimiser = Optimiser(model, error, learning_rule, train_data, valid_data, data_monitors)

stats3, keys3, run_time3 = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)

train_data.batch_size = batch_size 
valid_data.batch_size = batch_size

model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

error = CrossEntropySoftmaxError()

learning_rule = MomentumLearningRule(learning_rate=learning_rate, mom_coeff=mom_coeff)

_ = stats3, keys3, run_time3


# In[56]:

# Seed a random number generator
seed = 10102016 
rng = np.random.RandomState(seed)

# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# Create data provider objects for the MNIST data set
train_data = MNISTDataProvider('train', batch_size=50, rng=rng)
valid_data = MNISTDataProvider('valid', batch_size=50, rng=rng)

input_dim, output_dim, hidden_dim = 784, 10, 100

#Set training run hyperparameters
batch_size = 50  # number of data points in a batch
init_scale = 0.01  # scale for random parameter initialisation
num_epochs = 100  # number of training epochs to perform
learning_rate = 0.1

stats_interval = 5  # epoch interval between recording and printing stats
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)
data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

optimiser = Optimiser(model, error, learning_rule, train_data, valid_data, data_monitors)

stats4, keys4, run_time4 = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)

train_data.batch_size = batch_size 
valid_data.batch_size = batch_size

model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

error = CrossEntropySoftmaxError()

learning_rule = GradientDescentLearningRule(learning_rate=learning_rate)
_ = stats4, keys4, run_time4


# In[57]:

fig_1 = plt.figure(figsize=(8, 4))
ax_1 = fig_1.add_subplot(111)
for k in ['error(train)', 'error(valid)']:
    ax_1.plot(np.arange(1, stats1.shape[0]) * stats_interval, stats1[1:, keys1[k]], label=k+"mom_coeff=0.1")
    ax_1.plot(np.arange(1, stats2.shape[0]) * stats_interval, stats2[1:, keys2[k]], label=k+"mom_coeff=0.5")
    ax_1.plot(np.arange(1, stats3.shape[0]) * stats_interval, stats3[1:, keys3[k]], label=k+"mom_coeff=0.9")
    ax_1.plot(np.arange(1, stats4.shape[0]) * stats_interval, stats4[1:, keys4[k]], label=k+"basic rule")
ax_1.legend(loc=0)
ax_1.set_xlabel('Epoch number')


fig_2 = plt.figure(figsize=(8, 4))
ax_2 = fig_2.add_subplot(111)
for k in ['acc(train)', 'acc(valid)']:
    ax_2.plot(np.arange(1, stats1.shape[0]) * stats_interval, stats1[1:, keys1[k]], label=k+"mom_coeff=0.1")
    ax_2.plot(np.arange(1, stats2.shape[0]) * stats_interval, stats2[1:, keys2[k]], label=k+"mom_coeff=0.5")
    ax_2.plot(np.arange(1, stats3.shape[0]) * stats_interval, stats3[1:, keys3[k]], label=k+"mom_coeff=0.9")
    ax_2.plot(np.arange(1, stats4.shape[0]) * stats_interval, stats4[1:, keys4[k]], label=k+"basic rule")
ax_2.legend(loc=0)
ax_2.set_xlabel('Epoch number')
    
_=fig_1, ax_1, fig_2, ax_2
    
fig_1.savefig('7.pdf')
fig_2.savefig('8.pdf')
fig_1.savefig('7.svg')
fig_2.savefig('8.svg')


# # Using MomentumLearningRule with constant mom_coeff

# In[10]:

# The model set up code below is provided as a starting point.
# You will probably want to add further code cells for the
# different experiments you run.
import matplotlib.pyplot as plt
from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import GradientDescentLearningRule
from mlp.learning_rules import MomentumLearningRule
from mlp.optimisers import Optimiser
get_ipython().magic(u'matplotlib inline')
plt.style.use('ggplot')

batch_size = 50  # number of data points in a batch
init_scale = 0.1  # scale for random parameter initialisation
num_epochs = 100  # number of training epochs to perform
learning_rate = 0.1
mom_coeff = 0.9
stats_interval = 5  # epoch interval between recording and printing stats


input_dim, output_dim, hidden_dim = 784, 10, 100

weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

error = CrossEntropySoftmaxError()

learning_rule = MomentumLearningRule(learning_rate=learning_rate, mom_coeff=mom_coeff)

_ = train_model_and_plot_stats(
    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval)


# Using GradientDescentLearningRule

# In[11]:

import matplotlib.pyplot as plt
from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import GradientDescentLearningRule
from mlp.schedulers import TimedependentLearningRateScheduler
from mlp.optimisers import Optimiser
from mlp.schedulers import ConstantLearningRateScheduler
get_ipython().magic(u'matplotlib inline')
plt.style.use('ggplot')


input_dim, output_dim, hidden_dim = 784, 10, 100

#Set training run hyperparameters
batch_size = 50  # number of data points in a batch
init_scale = 0.1  # scale for random parameter initialisation
num_epochs = 100  # number of training epochs to perform
learning_rate = 0.1
stats_interval = 5  # epoch interval between recording and printing stats
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

train_data.batch_size = batch_size 
valid_data.batch_size = batch_size

model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

error = CrossEntropySoftmaxError()

learning_rule = GradientDescentLearningRule(learning_rate=learning_rate)

_ = train_model_and_plot_stats(
    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval)


# Using momentum coeffcient scheduler

# In[6]:

import numpy as np
import logging
from mlp.data_providers import MNISTDataProvider
import matplotlib.pyplot as plt
from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import GradientDescentLearningRule
from mlp.schedulers import TimedependentLearningRateScheduler
from mlp.optimisers import Optimiser
from mlp.schedulers import ConstantLearningRateScheduler
import matplotlib.pyplot as plt
from mlp.learning_rules import MomentumLearningRule
from mlp.schedulers import MomentumLearningRuleScheduler

get_ipython().magic(u'matplotlib inline')
plt.style.use('ggplot')


# In[9]:

# Seed a random number generator
seed = 10102016 
rng = np.random.RandomState(seed)

# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# Create data provider objects for the MNIST data set
train_data = MNISTDataProvider('train', batch_size=50, rng=rng)
valid_data = MNISTDataProvider('valid', batch_size=50, rng=rng)

input_dim, output_dim, hidden_dim = 784, 10, 100

#Set training run hyperparameters
batch_size = 50  # number of data points in a batch
init_scale = 0.01  # scale for random parameter initialisation
num_epochs = 100  # number of training epochs to perform
learning_rate = 0.1

mom_coeff = 0.9
alpha_inf = 0.9
tau = 20
gamma = 6.7

stats_interval = 5  # epoch interval between recording and printing stats
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)
data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

error = CrossEntropySoftmaxError()

learning_rule = MomentumLearningRule(learning_rate=learning_rate, mom_coeff=mom_coeff)

schedulers = [MomentumLearningRuleScheduler(learning_rate=learning_rate,alpha_inf=alpha_inf,gamma=gamma,tau=tau)]

train_data.batch_size = batch_size 
valid_data.batch_size = batch_size

optimiser = Optimiser(model, error, learning_rule, train_data, valid_data, data_monitors,schedulers)

stats1, keys1, run_time1 = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)

_ = stats1, keys1, run_time1


# In[10]:

# Seed a random number generator
seed = 10102016 
rng = np.random.RandomState(seed)

# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# Create data provider objects for the MNIST data set
train_data = MNISTDataProvider('train', batch_size=50, rng=rng)
valid_data = MNISTDataProvider('valid', batch_size=50, rng=rng)

input_dim, output_dim, hidden_dim = 784, 10, 100

#Set training run hyperparameters
batch_size = 50  # number of data points in a batch
init_scale = 0.01  # scale for random parameter initialisation
num_epochs = 100  # number of training epochs to perform
learning_rate = 0.1

mom_coeff = 0.9
alpha_inf = 0.9
tau = 50
gamma = 16.7

stats_interval = 5  # epoch interval between recording and printing stats
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)
data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

optimiser = Optimiser(model, error, learning_rule, train_data, valid_data, data_monitors,schedulers)

stats2, keys2, run_time2 = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)

train_data.batch_size = batch_size 
valid_data.batch_size = batch_size

model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

error = CrossEntropySoftmaxError()

learning_rule = MomentumLearningRule(learning_rate=learning_rate, mom_coeff=mom_coeff)

schedulers = [MomentumLearningRuleScheduler(learning_rate=learning_rate,alpha_inf=alpha_inf,gamma=gamma,tau=tau)]

_ = stats2, keys2, run_time2


# In[11]:

# Seed a random number generator
seed = 10102016 
rng = np.random.RandomState(seed)

# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# Create data provider objects for the MNIST data set
train_data = MNISTDataProvider('train', batch_size=50, rng=rng)
valid_data = MNISTDataProvider('valid', batch_size=50, rng=rng)

input_dim, output_dim, hidden_dim = 784, 10, 100

#Set training run hyperparameters
batch_size = 50  # number of data points in a batch
init_scale = 0.01  # scale for random parameter initialisation
num_epochs = 100  # number of training epochs to perform
learning_rate = 0.1

mom_coeff = 0.9
alpha_inf = 0.9
tau = 70
gamma = 23

stats_interval = 5  # epoch interval between recording and printing stats
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)
data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

optimiser = Optimiser(model, error, learning_rule, train_data, valid_data, data_monitors,schedulers)

stats3, keys3, run_time3 = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)

train_data.batch_size = batch_size 
valid_data.batch_size = batch_size

model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

error = CrossEntropySoftmaxError()

learning_rule = MomentumLearningRule(learning_rate=learning_rate, mom_coeff=mom_coeff)

schedulers = [MomentumLearningRuleScheduler(learning_rate=learning_rate,alpha_inf=alpha_inf,gamma=gamma,tau=tau)]

_ = stats3, keys3, run_time3


# In[7]:

# Seed a random number generator
seed = 10102016 
rng = np.random.RandomState(seed)

# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# Create data provider objects for the MNIST data set
train_data = MNISTDataProvider('train', batch_size=50, rng=rng)
valid_data = MNISTDataProvider('valid', batch_size=50, rng=rng)

input_dim, output_dim, hidden_dim = 784, 10, 100

#Set training run hyperparameters
batch_size = 50  # number of data points in a batch
init_scale = 0.01  # scale for random parameter initialisation
num_epochs = 100  # number of training epochs to perform
learning_rate = 0.1
mom_coeff = 0.9

stats_interval = 5  # epoch interval between recording and printing stats
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)
data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

optimiser = Optimiser(model, error, learning_rule, train_data, valid_data, data_monitors)

stats4, keys4, run_time4 = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)

train_data.batch_size = batch_size 
valid_data.batch_size = batch_size

model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

error = CrossEntropySoftmaxError()

learning_rule = MomentumLearningRule(learning_rate=learning_rate, mom_coeff=mom_coeff)

_ = stats4, keys4, run_time4


# In[12]:

fig_1 = plt.figure(figsize=(8, 4))
ax_1 = fig_1.add_subplot(111)
for k in ['error(train)', 'error(valid)']:
    ax_1.plot(np.arange(1, stats1.shape[0]) * stats_interval, stats1[1:, keys1[k]], label=k+"gamma=6.7,tau=20")
    ax_1.plot(np.arange(1, stats2.shape[0]) * stats_interval, stats2[1:, keys2[k]], label=k+"gamma=16.7,tau=50")
    ax_1.plot(np.arange(1, stats3.shape[0]) * stats_interval, stats3[1:, keys3[k]], label=k+"gamma=23,tau=70")
    ax_1.plot(np.arange(1, stats4.shape[0]) * stats_interval, stats4[1:, keys4[k]], label=k+"mom basic line")
ax_1.legend(loc=0)
ax_1.set_xlabel('Epoch number')


fig_2 = plt.figure(figsize=(8, 4))
ax_2 = fig_2.add_subplot(111)
for k in ['acc(train)', 'acc(valid)']:
    ax_2.plot(np.arange(1, stats1.shape[0]) * stats_interval, stats1[1:, keys1[k]], label=k+"gamma=6.7,tau=20")
    ax_2.plot(np.arange(1, stats2.shape[0]) * stats_interval, stats2[1:, keys2[k]], label=k+"gamma=16.7,tau=50")
    ax_2.plot(np.arange(1, stats3.shape[0]) * stats_interval, stats3[1:, keys3[k]], label=k+"gamma=23,tau=70")
    ax_1.plot(np.arange(1, stats4.shape[0]) * stats_interval, stats4[1:, keys4[k]], label=k+"mom basic line")
ax_2.legend(loc=0)
ax_2.set_xlabel('Epoch number')
    
_=fig_1, ax_1, fig_2, ax_2
    
fig_1.savefig('20.pdf')
fig_2.savefig('21.pdf')
fig_1.savefig('20.svg')
fig_2.savefig('21.svg')


# In[ ]:




# In[ ]:




# In[7]:

import matplotlib.pyplot as plt
from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import GradientDescentLearningRule,MomentumLearningRule
from mlp.schedulers import TimedependentLearningRateScheduler
from mlp.optimisers import Optimiser
from mlp.schedulers import ConstantLearningRateScheduler
from mlp.schedulers import MomentumLearningRuleScheduler
get_ipython().magic(u'matplotlib inline')
plt.style.use('ggplot')


input_dim, output_dim, hidden_dim = 784, 10, 100

#Set training run hyperparameters
batch_size = 50  # number of data points in a batch
init_scale = 0.1  # scale for random parameter initialisation
num_epochs = 100  # number of training epochs to perform
learning_rate = 0.1
mom_coeff = 0.9
alpha_inf = 0.1
tau = 50
gamma = 50
stats_interval = 5  # epoch interval between recording and printing stats
weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

train_data.batch_size = batch_size 
valid_data.batch_size = batch_size

model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

error = CrossEntropySoftmaxError()

learning_rule = MomentumLearningRule(learning_rate=learning_rate, mom_coeff=mom_coeff)

schedulers = [MomentumLearningRuleScheduler(learning_rate=learning_rate,alpha_inf=alpha_inf,gamma=gamma,tau=tau)]

_ = train_model_and_plot_stats_mom(
    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval)


# ## Part 3: Adaptive learning rules

# In[4]:

# The below code will set up the data providers, random number
# generator and logger objects needed for training runs. As
# loading the data from file take a little while you generally
# will probably not want to reload the data providers on
# every training run. If you wish to reset their state you
# should instead use the .reset() method of the data providers.
import numpy as np
import logging
from mlp.data_providers import MNISTDataProvider

# Seed a random number generator
seed = 10102016 
rng = np.random.RandomState(seed)

# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# Create data provider objects for the MNIST data set
train_data = MNISTDataProvider('train', batch_size=50, rng=rng)
valid_data = MNISTDataProvider('valid', batch_size=50, rng=rng)


# In[5]:

def train_model_and_plot_stats(
        model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval):

    # As well as monitoring the error over training also monitor classification
    # accuracy i.e. proportion of most-probable predicted classes being equal to targets
    data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

    # Use the created objects to initialise a new Optimiser instance.
   
    optimiser = Optimiser(
        model, error, learning_rule, train_data, valid_data, data_monitors)

    # Run the optimiser for 5 epochs (full passes through the training set)
    # printing statistics every epoch.
    stats, keys, run_time = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)

    # Plot the change in the validation and training set error over training.
    fig_1 = plt.figure(figsize=(8, 4))
    ax_1 = fig_1.add_subplot(111)
    for k in ['error(train)', 'error(valid)']:
        ax_1.plot(np.arange(1, stats.shape[0]) * stats_interval, 
                  stats[1:, keys[k]], label=k)
    ax_1.legend(loc=0)
    ax_1.set_xlabel('Epoch number')

    # Plot the change in the validation and training set accuracy over training.
    fig_2 = plt.figure(figsize=(8, 4))
    ax_2 = fig_2.add_subplot(111)
    for k in ['acc(train)', 'acc(valid)']:
        ax_2.plot(np.arange(1, stats.shape[0]) * stats_interval, 
                  stats[1:, keys[k]], label=k)
    ax_2.legend(loc=0)
    ax_2.set_xlabel('Epoch number')
    
    return stats, keys, run_time, fig_1, ax_1, fig_2, ax_2


# AdaGradLearningRule

# In[6]:

import matplotlib.pyplot as plt
from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import GradientDescentLearningRule,MomentumLearningRule,AdaGradLearningRule
from mlp.schedulers import TimedependentLearningRateScheduler
from mlp.optimisers import Optimiser 
from mlp.schedulers import ConstantLearningRateScheduler

get_ipython().magic(u'matplotlib inline')
plt.style.use('ggplot')


input_dim, output_dim, hidden_dim = 784, 10, 100

#Set training run hyperparameters
batch_size = 50  # number of data points in a batch
init_scale = 0.1  # scale for random parameter initialisation
num_epochs = 100  # number of training epochs to perform
learning_rate = 0.0001
stats_interval = 5  # epoch interval between recording and printing stats
e = 1e-8

weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

train_data.batch_size = batch_size 
valid_data.batch_size = batch_size

model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

error = CrossEntropySoftmaxError()

learning_rule = AdaGradLearningRule(learning_rate=learning_rate)

_ = train_model_and_plot_stats(
    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval)


# RMSpropLearningRule

# In[7]:

# The below code will set up the data providers, random number
# generator and logger objects needed for training runs. As
# loading the data from file take a little while you generally
# will probably not want to reload the data providers on
# every training run. If you wish to reset their state you
# should instead use the .reset() method of the data providers.
import numpy as np
import logging
from mlp.data_providers import MNISTDataProvider

# Seed a random number generator
seed = 10102016 
rng = np.random.RandomState(seed)

# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# Create data provider objects for the MNIST data set
train_data = MNISTDataProvider('train', batch_size=50, rng=rng)
valid_data = MNISTDataProvider('valid', batch_size=50, rng=rng)


# In[8]:

def train_model_and_plot_stats(
        model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval):

    # As well as monitoring the error over training also monitor classification
    # accuracy i.e. proportion of most-probable predicted classes being equal to targets
    data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

    # Use the created objects to initialise a new Optimiser instance.
   
    optimiser = Optimiser(
        model, error, learning_rule, train_data, valid_data, data_monitors)

    # Run the optimiser for 5 epochs (full passes through the training set)
    # printing statistics every epoch.
    stats, keys, run_time = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)

    # Plot the change in the validation and training set error over training.
    fig_1 = plt.figure(figsize=(8, 4))
    ax_1 = fig_1.add_subplot(111)
    for k in ['error(train)', 'error(valid)']:
        ax_1.plot(np.arange(1, stats.shape[0]) * stats_interval, 
                  stats[1:, keys[k]], label=k)
    ax_1.legend(loc=0)
    ax_1.set_xlabel('Epoch number')

    # Plot the change in the validation and training set accuracy over training.
    fig_2 = plt.figure(figsize=(8, 4))
    ax_2 = fig_2.add_subplot(111)
    for k in ['acc(train)', 'acc(valid)']:
        ax_2.plot(np.arange(1, stats.shape[0]) * stats_interval, 
                  stats[1:, keys[k]], label=k)
    ax_2.legend(loc=0)
    ax_2.set_xlabel('Epoch number')
    
    return stats, keys, run_time, fig_1, ax_1, fig_2, ax_2


# In[9]:

import matplotlib.pyplot as plt
from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import GradientDescentLearningRule,MomentumLearningRule,AdaGradLearningRule,RMSpropLearningRule
from mlp.schedulers import TimedependentLearningRateScheduler
from mlp.optimisers import Optimiser 
from mlp.schedulers import ConstantLearningRateScheduler
get_ipython().magic(u'matplotlib inline')
plt.style.use('ggplot')

input_dim, output_dim, hidden_dim = 784, 10, 100

#Set training run hyperparameters
batch_size = 50  # number of data points in a batch
init_scale = 0.1  # scale for random parameter initialisation
num_epochs = 100  # number of training epochs to perform
learning_rate = 0.0001
stats_interval = 5  # epoch interval between recording and printing stats
e = 1e-4

weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

train_data.batch_size = batch_size 
valid_data.batch_size = batch_size

model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

error = CrossEntropySoftmaxError()

learning_rule = RMSpropLearningRule(learning_rate=learning_rate,e=10**-8,beta=0.6)

_ = train_model_and_plot_stats(
    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval)


# In[1]:

import numpy as np
import logging
from mlp.data_providers import MNISTDataProvider

# Seed a random number generator
seed = 10102016 
rng = np.random.RandomState(seed)

# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# Create data provider objects for the MNIST data set
train_data = MNISTDataProvider('train', batch_size=50, rng=rng)
valid_data = MNISTDataProvider('valid', batch_size=50, rng=rng)


# In[2]:

def train_model_and_plot_stats(
        model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval):

    # As well as monitoring the error over training also monitor classification
    # accuracy i.e. proportion of most-probable predicted classes being equal to targets
    data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

    # Use the created objects to initialise a new Optimiser instance.
   
    optimiser = Optimiser(
        model, error, learning_rule, train_data, valid_data, data_monitors)

    # Run the optimiser for 5 epochs (full passes through the training set)
    # printing statistics every epoch.
    stats, keys, run_time = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)

    # Plot the change in the validation and training set error over training.
    fig_1 = plt.figure(figsize=(8, 4))
    ax_1 = fig_1.add_subplot(111)
    for k in ['error(train)', 'error(valid)']:
        ax_1.plot(np.arange(1, stats.shape[0]) * stats_interval, 
                  stats[1:, keys[k]], label=k)
    ax_1.legend(loc=0)
    ax_1.set_xlabel('Epoch number')

    # Plot the change in the validation and training set accuracy over training.
    fig_2 = plt.figure(figsize=(8, 4))
    ax_2 = fig_2.add_subplot(111)
    for k in ['acc(train)', 'acc(valid)']:
        ax_2.plot(np.arange(1, stats.shape[0]) * stats_interval, 
                  stats[1:, keys[k]], label=k)
    ax_2.legend(loc=0)
    ax_2.set_xlabel('Epoch number')
    
    return stats, keys, run_time, fig_1, ax_1, fig_2, ax_2


# In[3]:

import matplotlib.pyplot as plt
from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import GradientDescentLearningRule,MomentumLearningRule,AdaGradLearningRule,RMSpropLearningRule
from mlp.schedulers import TimedependentLearningRateScheduler
from mlp.optimisers import Optimiser 
from mlp.schedulers import ConstantLearningRateScheduler
from mlp.learning_rules import MomentumLearningRule
from mlp.schedulers import MomentumLearningRuleScheduler
get_ipython().magic(u'matplotlib inline')
plt.style.use('ggplot')

input_dim, output_dim, hidden_dim = 784, 10, 100

#Set training run hyperparameters
batch_size = 50  # number of data points in a batch
init_scale = 0.1  # scale for random parameter initialisation
num_epochs = 100  # number of training epochs to perform
learning_rate = 0.0001
stats_interval = 5  # epoch interval between recording and printing stats
mom_coeff = 0.5

weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

train_data.batch_size = batch_size 
valid_data.batch_size = batch_size

model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

error = CrossEntropySoftmaxError()

learning_rule = MomentumLearningRule(learning_rate=learning_rate, mom_coeff=mom_coeff)
_ = train_model_and_plot_stats(
    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval)


# basic

# In[4]:

import numpy as np
import logging
from mlp.data_providers import MNISTDataProvider

# Seed a random number generator
seed = 10102016 
rng = np.random.RandomState(seed)

# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# Create data provider objects for the MNIST data set
train_data = MNISTDataProvider('train', batch_size=50, rng=rng)
valid_data = MNISTDataProvider('valid', batch_size=50, rng=rng)


# In[5]:

def train_model_and_plot_stats(
        model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval):

    # As well as monitoring the error over training also monitor classification
    # accuracy i.e. proportion of most-probable predicted classes being equal to targets
    data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

    # Use the created objects to initialise a new Optimiser instance.
   
    optimiser = Optimiser(
        model, error, learning_rule, train_data, valid_data, data_monitors)

    # Run the optimiser for 5 epochs (full passes through the training set)
    # printing statistics every epoch.
    stats, keys, run_time = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)

    # Plot the change in the validation and training set error over training.
    fig_1 = plt.figure(figsize=(8, 4))
    ax_1 = fig_1.add_subplot(111)
    for k in ['error(train)', 'error(valid)']:
        ax_1.plot(np.arange(1, stats.shape[0]) * stats_interval, 
                  stats[1:, keys[k]], label=k)
    ax_1.legend(loc=0)
    ax_1.set_xlabel('Epoch number')

    # Plot the change in the validation and training set accuracy over training.
    fig_2 = plt.figure(figsize=(8, 4))
    ax_2 = fig_2.add_subplot(111)
    for k in ['acc(train)', 'acc(valid)']:
        ax_2.plot(np.arange(1, stats.shape[0]) * stats_interval, 
                  stats[1:, keys[k]], label=k)
    ax_2.legend(loc=0)
    ax_2.set_xlabel('Epoch number')
    
    return stats, keys, run_time, fig_1, ax_1, fig_2, ax_2


# In[5]:

import matplotlib.pyplot as plt
from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit
from mlp.learning_rules import GradientDescentLearningRule,MomentumLearningRule,AdaGradLearningRule,RMSpropLearningRule
from mlp.schedulers import TimedependentLearningRateScheduler
from mlp.optimisers import Optimiser 
from mlp.schedulers import ConstantLearningRateScheduler
from mlp.learning_rules import MomentumLearningRule
from mlp.schedulers import MomentumLearningRuleScheduler
get_ipython().magic(u'matplotlib inline')
plt.style.use('ggplot')

input_dim, output_dim, hidden_dim = 784, 10, 100

#Set training run hyperparameters
batch_size = 50  # number of data points in a batch
init_scale = 0.1  # scale for random parameter initialisation
num_epochs = 100  # number of training epochs to perform
learning_rate = 0.0001
stats_interval = 5  # epoch interval between recording and printing stats
e = 1e-6

weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

train_data.batch_size = batch_size 
valid_data.batch_size = batch_size

model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

error = CrossEntropySoftmaxError()

learning_rule = GradientDescentLearningRule(learning_rate=learning_rate)

_ = train_model_and_plot_stats(
    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval)

