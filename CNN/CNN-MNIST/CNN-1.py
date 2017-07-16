
# coding: utf-8

# In[3]:

import mlp.layers as layers
import mlp.initialisers as init
from scipy import signal

class ConvolutionalLayer(layers.LayerWithParameters):
    """Layer implementing a 2D convolution-based transformation of its inputs.

    The layer is parameterised by a set of 2D convolutional kernels, a four
    dimensional array of shape
        (num_output_channels, num_input_channels, kernel_dim_1, kernel_dim_2)
    and a bias vector, a one dimensional array of shape
        (num_output_channels,)
    i.e. one shared bias per output channel.

    Assuming no-padding is applied to the inputs so that outputs are only
    calculated for positions where the kernel filters fully overlap with the
    inputs, and that unit strides are used the outputs will have spatial extent
        output_dim_1 = input_dim_1 - kernel_dim_1 + 1
        output_dim_2 = input_dim_2 - kernel_dim_2 + 1
    """

    def __init__(self, num_input_channels, num_output_channels,
                 input_dim_1, input_dim_2,
                 kernel_dim_1, kernel_dim_2,
                 kernels_init=init.UniformInit(-0.01, 0.01),
                 biases_init=init.ConstantInit(0.),
                 kernels_penalty=None, biases_penalty=None):
        """Initialises a parameterised convolutional layer.

        Args:
            num_input_channels (int): Number of channels in inputs to
                layer (this may be number of colour channels in the input
                images if used as the first layer in a model, or the
                number of output channels, a.k.a. feature maps, from a
                a previous convolutional layer).
            num_output_channels (int): Number of channels in outputs
                from the layer, a.k.a. number of feature maps.
            input_dim_1 (int): Size of first input dimension of each 2D
                channel of inputs.
            input_dim_2 (int): Size of second input dimension of each 2D
                channel of inputs.
            kernel_dim_x (int): Size of first dimension of each 2D channel of
                kernels.
            kernel_dim_y (int): Size of second dimension of each 2D channel of
                kernels.
            kernels_intialiser: Initialiser for the kernel parameters.
            biases_initialiser: Initialiser for the bias parameters.
            kernels_penalty: Kernel-dependent penalty term (regulariser) or
                None if no regularisation is to be applied to the kernels.
            biases_penalty: Biases-dependent penalty term (regulariser) or
                None if no regularisation is to be applied to the biases.
        """
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.input_dim_1 = input_dim_1
        self.input_dim_2 = input_dim_2
        self.kernel_dim_1 = kernel_dim_1
        self.kernel_dim_2 = kernel_dim_2
        self.kernels_init = kernels_init
        self.biases_init = biases_init
        self.kernels_shape = (
            num_output_channels, num_input_channels, kernel_dim_1, kernel_dim_2
        )
        self.inputs_shape = (
            None, num_input_channels, input_dim_1, input_dim_2
        )
        self.kernels = self.kernels_init(self.kernels_shape)
        self.biases = self.biases_init(num_output_channels)
        self.kernels_penalty = kernels_penalty
        self.biases_penalty = biases_penalty

    def fprop(self, inputs):
        """Forward propagates activations through the layer transformation.

        For inputs `x`, outputs `y`, kernels `K` and biases `b` the layer
        corresponds to `y = conv2d(x, K) + b`.

        Args:
            inputs: Array of layer inputs of shape (batch_size, input_dim).

        Returns:
            outputs: Array of layer outputs of shape (batch_size, output_dim).
        """
        output_dim_1 = self.input_dim_1 - self.kernel_dim_1 + 1
        output_dim_2 = self.input_dim_2 - self.kernel_dim_2 + 1
        self.batch_size = inputs.shape[0]
        output_size = self.batch_size*self.num_output_channels*output_dim_1*output_dim_2
        output_shape =(self.batch_size,self.num_output_channels,output_dim_1,output_dim_2)
        outputs = np.zeros(output_size).reshape(output_shape)
        for i in range(self.batch_size):
            for j in range(self.num_output_channels):
                for k in range(self.num_input_channels):
                    outputs[i][j] += signal.convolve2d(inputs[i][k],self.kernels[j][k],mode='valid')
                outputs[i][j] += self.biases[j]
        
        return outputs        
    

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        """Back propagates gradients through a layer.

        Given gradients with respect to the outputs of the layer calculates the
        gradients with respect to the layer inputs.

        Args:
            inputs: Array of layer inputs of shape
                (batch_size, num_input_channels, input_dim_1, input_dim_2).
            outputs: Array of layer outputs calculated in forward pass of
                shape
                (batch_size, num_output_channels, output_dim_1, output_dim_2).
            grads_wrt_outputs: Array of gradients with respect to the layer
                outputs of shape
                (batch_size, num_output_channels, output_dim_1, output_dim_2).

        Returns:
            Array of gradients with respect to the layer inputs of shape
            (batch_size, input_dim).
        """
        grads_wrt_inputs = np.zeros(inputs.shape)
        self.kernels = self.kernels[:,:,::-1,::-1]
        self.batch_size = inputs.shape[0]
        for i in range(self.batch_size):
            for j in range(self.num_input_channels):
                 for k in range(self.num_output_channels):
                    grads_wrt_inputs[i][j] += signal.convolve2d(grads_wrt_outputs[i][k],self.kernels[k][j])   

        return grads_wrt_inputs
    
    
    def grads_wrt_params(self, inputs, grads_wrt_outputs):
        """Calculates gradients with respect to layer parameters.

        Args:
            inputs: array of inputs to layer of shape (batch_size, input_dim)
            grads_wrt_to_outputs: array of gradients with respect to the layer
                outputs of shape
                (batch_size, num_output-_channels, output_dim_1, output_dim_2).

        Returns:
            list of arrays of gradients with respect to the layer parameters
            `[grads_wrt_kernels, grads_wrt_biases]`.
        """
        kernel_grads = np.zeros(self.kernels.shape)
        bias_grads = np.zeros(self.biases.shape)
        inputs = inputs[:,:,::-1,::-1]
        self.batch_size = inputs.shape[0]
        for i in range(self.num_output_channels):
            for j in range(self.num_input_channels):
                for k in range(self.batch_size):
                    kernel_grads[i][j] += signal.convolve2d(inputs[k][j],grads_wrt_outputs[k][i],mode = 'valid')
                    bias_grads[i] += (np.sum(grads_wrt_outputs[k][i]))/(self.num_input_channels)
        if self.kernels_penalty is not None:
            kernel_grads += self.kernels_penalty.grad(self.kernels)
        if self.biases_penalty is not None:
            bias_grads += self.biases_penalty.grad(self.biases) 
        return [kernel_grads,bias_grads]    
                

    def params_penalty(self):
        """Returns the parameter dependent penalty term for this layer.

        If no parameter-dependent penalty terms are set this returns zero.
        """
        params_penalty = 0
        if self.kernels_penalty is not None:
            params_penalty += self.kernels_penalty(self.kernels)
        if self.biases_penalty is not None:
            params_penalty += self.biases_penalty(self.biases)
        return params_penalty

    @property
    def params(self):
        """A list of layer parameter values: `[kernels, biases]`."""
        return [self.kernels, self.biases]

    @params.setter
    def params(self, values):
        self.kernels = values[0]
        self.biases = values[1]

    def __repr__(self):
        return (
            'ConvolutionalLayer(\n'
            '    num_input_channels={0}, num_output_channels={1},\n'
            '    input_dim_1={2}, input_dim_2={3},\n'
            '    kernel_dim_1={4}, kernel_dim_2={5}\n'
            ')'
            .format(self.num_input_channels, self.num_output_channels,
                    self.input_dim_1, self.input_dim_2, self.kernel_dim_1,
                    self.kernel_dim_2)
        )


# In[2]:

import numpy as np

def test_conv_layer_fprop(layer_class, do_cross_correlation=False):
    """Tests `fprop` method of a convolutional layer.
    
    Checks the outputs of `fprop` method for a fixed input against known
    reference values for the outputs and raises an AssertionError if
    the outputted values are not consistent with the reference values. If
    tests are all passed returns True.
    
    Args:
        layer_class: Convolutional layer implementation following the 
            interface defined in the provided skeleton class.
        do_cross_correlation: Whether the layer implements an operation
            corresponding to cross-correlation (True) i.e kernels are
            not flipped before sliding over inputs, or convolution
            (False) with filters being flipped.

    Raises:
        AssertionError: Raised if output of `layer.fprop` is inconsistent 
            with reference values either in shape or values.
    """
    inputs = np.arange(96).reshape((2, 3, 4, 4))
    kernels = np.arange(-12, 12).reshape((2, 3, 2, 2))
    if do_cross_correlation:
        kernels = kernels[:, :, ::-1, ::-1]
    biases = np.arange(2)
    true_output = np.array(
        [[[[ -958., -1036., -1114.],
           [-1270., -1348., -1426.],
           [-1582., -1660., -1738.]],
          [[ 1707.,  1773.,  1839.],
           [ 1971.,  2037.,  2103.],
           [ 2235.,  2301.,  2367.]]],
         [[[-4702., -4780., -4858.],
           [-5014., -5092., -5170.],
           [-5326., -5404., -5482.]],
          [[ 4875.,  4941.,  5007.],
           [ 5139.,  5205.,  5271.],
           [ 5403.,  5469.,  5535.]]]]
    )
    layer = layer_class(
        num_input_channels=kernels.shape[1], 
        num_output_channels=kernels.shape[0], 
        input_dim_1=inputs.shape[2], 
        input_dim_2=inputs.shape[3],
        kernel_dim_1=kernels.shape[2],
        kernel_dim_2=kernels.shape[3]
    )
    layer.params = [kernels, biases]
    layer_output = layer.fprop(inputs)
    assert layer_output.shape == true_output.shape, (
        'Layer fprop gives incorrect shaped output. '
        'Correct shape is \n\n{0}\n\n but returned shape is \n\n{1}.'
        .format(true_output.shape, layer_output.shape)
    )
    assert np.allclose(layer_output, true_output), (
        'Layer fprop does not give correct output. '
        'Correct output is \n\n{0}\n\n but returned output is \n\n{1}.'
        .format(true_output, layer_output)
    )
    return True

def test_conv_layer_bprop(layer_class, do_cross_correlation=False):
    """Tests `bprop` method of a convolutional layer.
    
    Checks the outputs of `bprop` method for a fixed input against known
    reference values for the gradients with respect to inputs and raises 
    an AssertionError if the returned values are not consistent with the
    reference values. If tests are all passed returns True.
    
    Args:
        layer_class: Convolutional layer implementation following the 
            interface defined in the provided skeleton class.
        do_cross_correlation: Whether the layer implements an operation
            corresponding to cross-correlation (True) i.e kernels are
            not flipped before sliding over inputs, or convolution
            (False) with filters being flipped.

    Raises:
        AssertionError: Raised if output of `layer.bprop` is inconsistent 
            with reference values either in shape or values.
    """
    inputs = np.arange(96).reshape((2, 3, 4, 4))
    kernels = np.arange(-12, 12).reshape((2, 3, 2, 2))
    if do_cross_correlation:
        kernels = kernels[:, :, ::-1, ::-1]
    biases = np.arange(2)
    grads_wrt_outputs = np.arange(-20, 16).reshape((2, 2, 3, 3))
    outputs = np.array(
        [[[[ -958., -1036., -1114.],
           [-1270., -1348., -1426.],
           [-1582., -1660., -1738.]],
          [[ 1707.,  1773.,  1839.],
           [ 1971.,  2037.,  2103.],
           [ 2235.,  2301.,  2367.]]],
         [[[-4702., -4780., -4858.],
           [-5014., -5092., -5170.],
           [-5326., -5404., -5482.]],
          [[ 4875.,  4941.,  5007.],
           [ 5139.,  5205.,  5271.],
           [ 5403.,  5469.,  5535.]]]]
    )
    true_grads_wrt_inputs = np.array(
      [[[[ 147.,  319.,  305.,  162.],
         [ 338.,  716.,  680.,  354.],
         [ 290.,  608.,  572.,  294.],
         [ 149.,  307.,  285.,  144.]],
        [[  23.,   79.,   81.,   54.],
         [ 114.,  284.,  280.,  162.],
         [ 114.,  272.,  268.,  150.],
         [  73.,  163.,  157.,   84.]],
        [[-101., -161., -143.,  -54.],
         [-110., -148., -120.,  -30.],
         [ -62.,  -64.,  -36.,    6.],
         [  -3.,   19.,   29.,   24.]]],
       [[[  39.,   67.,   53.,   18.],
         [  50.,   68.,   32.,   -6.],
         [   2.,  -40.,  -76.,  -66.],
         [ -31.,  -89., -111.,  -72.]],
        [[  59.,  115.,  117.,   54.],
         [ 114.,  212.,  208.,   90.],
         [ 114.,  200.,  196.,   78.],
         [  37.,   55.,   49.,   12.]],
        [[  79.,  163.,  181.,   90.],
         [ 178.,  356.,  384.,  186.],
         [ 226.,  440.,  468.,  222.],
         [ 105.,  199.,  209.,   96.]]]])
    layer = layer_class(
    num_input_channels=kernels.shape[1],
    num_output_channels=kernels.shape[0],
    input_dim_1=inputs.shape[2],
    input_dim_2=inputs.shape[3],
    kernel_dim_1=kernels.shape[2],
    kernel_dim_2=kernels.shape[3]
    )
    layer.params = [kernels, biases]
    layer_grads_wrt_inputs = layer.bprop(inputs, outputs, grads_wrt_outputs)
    assert layer_grads_wrt_inputs.shape == true_grads_wrt_inputs.shape, (
        'Layer bprop returns incorrect shaped array. '
        'Correct shape is \n\n{0}\n\n but returned shape is \n\n{1}.'
        .format(true_grads_wrt_inputs.shape, layer_grads_wrt_inputs.shape)
    )
    assert np.allclose(layer_grads_wrt_inputs, true_grads_wrt_inputs), (
        'Layer bprop does not return correct values. '
        'Correct output is \n\n{0}\n\n but returned output is \n\n{1}'
        .format(true_grads_wrt_inputs, layer_grads_wrt_inputs)
    )
    return True

def test_conv_layer_grad_wrt_params(
        layer_class, do_cross_correlation=False):
    """Tests `grad_wrt_params` method of a convolutional layer.
    
    Checks the outputs of `grad_wrt_params` method for fixed inputs 
    against known reference values for the gradients with respect to 
    kernels and biases, and raises an AssertionError if the returned
    values are not consistent with the reference values. If tests
    are all passed returns True.
    
    Args:
        layer_class: Convolutional layer implementation following the 
            interface defined in the provided skeleton class.
        do_cross_correlation: Whether the layer implements an operation
            corresponding to cross-correlation (True) i.e kernels are
            not flipped before sliding over inputs, or convolution
            (False) with filters being flipped.

    Raises:
        AssertionError: Raised if output of `layer.bprop` is inconsistent 
            with reference values either in shape or values.
    """
    inputs = np.arange(96).reshape((2, 3, 4, 4))
    kernels = np.arange(-12, 12).reshape((2, 3, 2, 2))
    biases = np.arange(2)
    grads_wrt_outputs = np.arange(-20, 16).reshape((2, 2, 3, 3))
    true_kernel_grads = np.array(
        [[[[ -240.,  -114.],
         [  264.,   390.]],
        [[-2256., -2130.],
         [-1752., -1626.]],
        [[-4272., -4146.],
         [-3768., -3642.]]],
       [[[ 5268.,  5232.],
         [ 5124.,  5088.]],
        [[ 5844.,  5808.],
         [ 5700.,  5664.]],
        [[ 6420.,  6384.],
         [ 6276.,  6240.]]]])
    if do_cross_correlation:
        kernels = kernels[:, :, ::-1, ::-1]
        true_kernel_grads = true_kernel_grads[:, :, ::-1, ::-1]
    true_bias_grads = np.array([-126.,   36.])
    layer = layer_class(
        num_input_channels=kernels.shape[1], 
        num_output_channels=kernels.shape[0], 
        input_dim_1=inputs.shape[2], 
        input_dim_2=inputs.shape[3],
        kernel_dim_1=kernels.shape[2],
        kernel_dim_2=kernels.shape[3]
    )
    layer.params = [kernels, biases]
    layer_kernel_grads, layer_bias_grads = (
        layer.grads_wrt_params(inputs, grads_wrt_outputs))
    assert layer_kernel_grads.shape == true_kernel_grads.shape, (
        'grads_wrt_params gives incorrect shaped kernel gradients output. '
        'Correct shape is \n\n{0}\n\n but returned shape is \n\n{1}.'
        .format(true_kernel_grads.shape, layer_kernel_grads.shape)
    )
    assert np.allclose(layer_kernel_grads, true_kernel_grads), (
        'grads_wrt_params does not give correct kernel gradients output. '
        'Correct output is \n\n{0}\n\n but returned output is \n\n{1}.'
        .format(true_kernel_grads, layer_kernel_grads)
    )
    assert layer_bias_grads.shape == true_bias_grads.shape, (
        'grads_wrt_params gives incorrect shaped bias gradients output. '
        'Correct shape is \n\n{0}\n\n but returned shape is \n\n{1}.'
        .format(true_bias_grads.shape, layer_bias_grads.shape)
    )
    assert np.allclose(layer_bias_grads, true_bias_grads), (
        'grads_wrt_params does not give correct bias gradients output. '
        'Correct output is \n\n{0}\n\n but returned output is \n\n{1}.'
        .format(true_bias_grads, layer_bias_grads)
    )
    return True


# In[5]:

import numpy as np

def test_conv_layer_fprop(layer_class, do_cross_correlation=False):
    """Tests `fprop` method of a convolutional layer.
    
    Checks the outputs of `fprop` method for a fixed input against known
    reference values for the outputs and raises an AssertionError if
    the outputted values are not consistent with the reference values. If
    tests are all passed returns True.
    
    Args:
        layer_class: Convolutional layer implementation following the 
            interface defined in the provided skeleton class.
        do_cross_correlation: Whether the layer implements an operation
            corresponding to cross-correlation (True) i.e kernels are
            not flipped before sliding over inputs, or convolution
            (False) with filters being flipped.

    Raises:
        AssertionError: Raised if output of `layer.fprop` is inconsistent 
            with reference values either in shape or values.
    """
    inputs = np.arange(96).reshape((2, 3, 4, 4))
    kernels = np.arange(-12, 12).reshape((2, 3, 2, 2))
    if do_cross_correlation:
        kernels = kernels[:, :, ::-1, ::-1]
    biases = np.arange(2)
    true_output = np.array(
        [[[[ -958., -1036., -1114.],
           [-1270., -1348., -1426.],
           [-1582., -1660., -1738.]],
          [[ 1707.,  1773.,  1839.],
           [ 1971.,  2037.,  2103.],
           [ 2235.,  2301.,  2367.]]],
         [[[-4702., -4780., -4858.],
           [-5014., -5092., -5170.],
           [-5326., -5404., -5482.]],
          [[ 4875.,  4941.,  5007.],
           [ 5139.,  5205.,  5271.],
           [ 5403.,  5469.,  5535.]]]]
    )
    layer = layer_class(
        num_input_channels=kernels.shape[1], 
        num_output_channels=kernels.shape[0], 
        input_dim_1=inputs.shape[2], 
        input_dim_2=inputs.shape[3],
        kernel_dim_1=kernels.shape[2],
        kernel_dim_2=kernels.shape[3]
    )
    layer.params = [kernels, biases]
    layer_output = layer.fprop(inputs)
    assert layer_output.shape == true_output.shape, (
        'Layer fprop gives incorrect shaped output. '
        'Correct shape is \n\n{0}\n\n but returned shape is \n\n{1}.'
        .format(true_output.shape, layer_output.shape)
    )
    assert np.allclose(layer_output, true_output), (
        'Layer fprop does not give correct output. '
        'Correct output is \n\n{0}\n\n but returned output is \n\n{1}.'
        .format(true_output, layer_output)
    )
    return True

def test_conv_layer_bprop(layer_class, do_cross_correlation=False):
    """Tests `bprop` method of a convolutional layer.
    
    Checks the outputs of `bprop` method for a fixed input against known
    reference values for the gradients with respect to inputs and raises 
    an AssertionError if the returned values are not consistent with the
    reference values. If tests are all passed returns True.
    
    Args:
        layer_class: Convolutional layer implementation following the 
            interface defined in the provided skeleton class.
        do_cross_correlation: Whether the layer implements an operation
            corresponding to cross-correlation (True) i.e kernels are
            not flipped before sliding over inputs, or convolution
            (False) with filters being flipped.

    Raises:
        AssertionError: Raised if output of `layer.bprop` is inconsistent 
            with reference values either in shape or values.
    """
    inputs = np.arange(96).reshape((2, 3, 4, 4))
    kernels = np.arange(-12, 12).reshape((2, 3, 2, 2))
    if do_cross_correlation:
        kernels = kernels[:, :, ::-1, ::-1]
    biases = np.arange(2)
    grads_wrt_outputs = np.arange(-20, 16).reshape((2, 2, 3, 3))
    outputs = np.array(
        [[[[ -958., -1036., -1114.],
           [-1270., -1348., -1426.],
           [-1582., -1660., -1738.]],
          [[ 1707.,  1773.,  1839.],
           [ 1971.,  2037.,  2103.],
           [ 2235.,  2301.,  2367.]]],
         [[[-4702., -4780., -4858.],
           [-5014., -5092., -5170.],
           [-5326., -5404., -5482.]],
          [[ 4875.,  4941.,  5007.],
           [ 5139.,  5205.,  5271.],
           [ 5403.,  5469.,  5535.]]]]
    )
    true_grads_wrt_inputs = np.array(
      [[[[ 147.,  319.,  305.,  162.],
         [ 338.,  716.,  680.,  354.],
         [ 290.,  608.,  572.,  294.],
         [ 149.,  307.,  285.,  144.]],
        [[  23.,   79.,   81.,   54.],
         [ 114.,  284.,  280.,  162.],
         [ 114.,  272.,  268.,  150.],
         [  73.,  163.,  157.,   84.]],
        [[-101., -161., -143.,  -54.],
         [-110., -148., -120.,  -30.],
         [ -62.,  -64.,  -36.,    6.],
         [  -3.,   19.,   29.,   24.]]],
       [[[  39.,   67.,   53.,   18.],
         [  50.,   68.,   32.,   -6.],
         [   2.,  -40.,  -76.,  -66.],
         [ -31.,  -89., -111.,  -72.]],
        [[  59.,  115.,  117.,   54.],
         [ 114.,  212.,  208.,   90.],
         [ 114.,  200.,  196.,   78.],
         [  37.,   55.,   49.,   12.]],
        [[  79.,  163.,  181.,   90.],
         [ 178.,  356.,  384.,  186.],
         [ 226.,  440.,  468.,  222.],
         [ 105.,  199.,  209.,   96.]]]])
    layer = layer_class(
    num_input_channels=kernels.shape[1],
    num_output_channels=kernels.shape[0],
    input_dim_1=inputs.shape[2],
    input_dim_2=inputs.shape[3],
    kernel_dim_1=kernels.shape[2],
    kernel_dim_2=kernels.shape[3]
    )
    layer.params = [kernels, biases]
    layer_grads_wrt_inputs = layer.bprop(inputs, outputs, grads_wrt_outputs)
    assert layer_grads_wrt_inputs.shape == true_grads_wrt_inputs.shape, (
        'Layer bprop returns incorrect shaped array. '
        'Correct shape is \n\n{0}\n\n but returned shape is \n\n{1}.'
        .format(true_grads_wrt_inputs.shape, layer_grads_wrt_inputs.shape)
    )
    assert np.allclose(layer_grads_wrt_inputs, true_grads_wrt_inputs), (
        'Layer bprop does not return correct values. '
        'Correct output is \n\n{0}\n\n but returned output is \n\n{1}'
        .format(true_grads_wrt_inputs, layer_grads_wrt_inputs)
    )
    return True

def test_conv_layer_grad_wrt_params(
        layer_class, do_cross_correlation=False):
    """Tests `grad_wrt_params` method of a convolutional layer.
    
    Checks the outputs of `grad_wrt_params` method for fixed inputs 
    against known reference values for the gradients with respect to 
    kernels and biases, and raises an AssertionError if the returned
    values are not consistent with the reference values. If tests
    are all passed returns True.
    
    Args:
        layer_class: Convolutional layer implementation following the 
            interface defined in the provided skeleton class.
        do_cross_correlation: Whether the layer implements an operation
            corresponding to cross-correlation (True) i.e kernels are
            not flipped before sliding over inputs, or convolution
            (False) with filters being flipped.

    Raises:
        AssertionError: Raised if output of `layer.bprop` is inconsistent 
            with reference values either in shape or values.
    """
    inputs = np.arange(96).reshape((2, 3, 4, 4))
    kernels = np.arange(-12, 12).reshape((2, 3, 2, 2))
    biases = np.arange(2)
    grads_wrt_outputs = np.arange(-20, 16).reshape((2, 2, 3, 3))
    true_kernel_grads = np.array(
        [[[[ -240.,  -114.],
         [  264.,   390.]],
        [[-2256., -2130.],
         [-1752., -1626.]],
        [[-4272., -4146.],
         [-3768., -3642.]]],
       [[[ 5268.,  5232.],
         [ 5124.,  5088.]],
        [[ 5844.,  5808.],
         [ 5700.,  5664.]],
        [[ 6420.,  6384.],
         [ 6276.,  6240.]]]])
    if do_cross_correlation:
        kernels = kernels[:, :, ::-1, ::-1]
        true_kernel_grads = true_kernel_grads[:, :, ::-1, ::-1]
    true_bias_grads = np.array([-126.,   36.])
    layer = layer_class(
        num_input_channels=kernels.shape[1], 
        num_output_channels=kernels.shape[0], 
        input_dim_1=inputs.shape[2], 
        input_dim_2=inputs.shape[3],
        kernel_dim_1=kernels.shape[2],
        kernel_dim_2=kernels.shape[3]
    )
    layer.params = [kernels, biases]
    layer_kernel_grads, layer_bias_grads = (
        layer.grads_wrt_params(inputs, grads_wrt_outputs))
    assert layer_kernel_grads.shape == true_kernel_grads.shape, (
        'grads_wrt_params gives incorrect shaped kernel gradients output. '
        'Correct shape is \n\n{0}\n\n but returned shape is \n\n{1}.'
        .format(true_kernel_grads.shape, layer_kernel_grads.shape)
    )
    assert np.allclose(layer_kernel_grads, true_kernel_grads), (
        'grads_wrt_params does not give correct kernel gradients output. '
        'Correct output is \n\n{0}\n\n but returned output is \n\n{1}.'
        .format(true_kernel_grads, layer_kernel_grads)
    )
    assert layer_bias_grads.shape == true_bias_grads.shape, (
        'grads_wrt_params gives incorrect shaped bias gradients output. '
        'Correct shape is \n\n{0}\n\n but returned shape is \n\n{1}.'
        .format(true_bias_grads.shape, layer_bias_grads.shape)
    )
    assert np.allclose(layer_bias_grads, true_bias_grads), (
        'grads_wrt_params does not give correct bias gradients output. '
        'Correct output is \n\n{0}\n\n but returned output is \n\n{1}.'
        .format(true_bias_grads, layer_bias_grads)
    )
    return True


# In[6]:

all_correct = test_conv_layer_fprop(ConvolutionalLayer, False)
all_correct &= test_conv_layer_bprop(ConvolutionalLayer, False)
all_correct &= test_conv_layer_grad_wrt_params(ConvolutionalLayer, False)
if all_correct:
    print('All tests passed.')


# baseline

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
num_epochs = 50  # number of training epochs to perform
learning_rate = 0.01
mom_coeff = 0.9
stats_interval = 5  # epoch interval between recording and printing stats

weights_init = GlorotUniformInit(rng=rng)
biases_init = ConstantInit(0.)

model = MultipleLayerModel([
    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
    SigmoidLayer(),
    AffineLayer(hidden_dim, output_dim, weights_init, biases_init)
])

data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

optimiser = Optimiser(model, error, learning_rule, train_data, valid_data, data_monitors)

rng.seed(seed)
train_data.reset()
valid_data.reset()

train_data.batch_size = batch_size 
valid_data.batch_size = batch_size

error = CrossEntropySoftmaxError()

learning_rule = MomentumLearningRule(learning_rate=learning_rate, mom_coeff=mom_coeff)

stats3, keys3, run_time3 = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)


# In[8]:

# Seed a random number generator
seed = 24102016 
rng = np.random.RandomState(seed)

num_epochs = 50
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
num_input_channels=1 
num_output_channels=3
input_dim_1=28
input_dim_2=28
kernel_dim_1=5
kernel_dim_2=5
output_dim_1 = input_dim_1 - kernel_dim_1 + 1
output_dim_2 = input_dim_2 - kernel_dim_2 + 1

weights_init = GlorotUniformInit(weights_init_gain, rng)
biases_init = ConstantInit(biases_init)
error = CrossEntropySoftmaxError()
learning_rule =MomentumLearningRule(learning_rate,mom_coeff)
data_monitors = {'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}
weights_penalties = 0

rng.seed(seed)
train_data.reset()
valid_data.reset()

model = MultipleLayerModel([
             ReshapeLayer((num_input_channels,input_dim_1,input_dim_2)),
             ConvolutionalLayer(num_input_channels, num_output_channels,input_dim_1, input_dim_2,kernel_dim_1, kernel_dim_2), 
             ReluLayer(),
             ReshapeLayer((num_output_channels*output_dim_1*output_dim_1,)),
             MaxPoolingLayer(4),     
             AffineLayer(3*6*24,hidden_dim,weights_init,biases_init),
             ReluLayer(),
             AffineLayer(hidden_dim,hidden_dim,weights_init,biases_init),
             ReluLayer(),
             AffineLayer(hidden_dim,output_dim,weights_init,biases_init)
         ])

optimiser = Optimiser(model, error, learning_rule, train_data, valid_data, data_monitors)
stats1,key1,run_time1 = optimiser.train(num_epochs, stats_interval=stats_interval)


# In[22]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.style.use('ggplot')

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

list = ['cnn','baseline']

for k in ['acc(train)',]:
    ax1.plot(np.arange(1, stats1.shape[0]) * stats_interval, 
                stats1[1:, key1[k]], label='cnn')
    ax1.plot(np.arange(1, stats3.shape[0]) * stats_interval, 
            stats3[1:, keys3[k]], label='baseline')

for k in ['acc(valid)',]:
    ax2.plot(np.arange(1, stats1.shape[0]) * stats_interval, 
                stats1[1:, key1[k]], label='cnn')
    ax2.plot(np.arange(1, stats3.shape[0]) * stats_interval, 
            stats3[1:, keys3[k]], label='baseline')
    
ax1.legend(loc=0)
ax1.set_xlabel('Epoch number')
ax1.set_ylabel('Training set accuracy')
ax2.legend(loc=0)
ax2.set_xlabel('Epoch number')
ax2.set_ylabel('Validation set accuracy')


# cnn without maxpooling

# In[27]:

# Seed a random number generator
seed = 24102016 
rng = np.random.RandomState(seed)

num_epochs = 50
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
num_input_channels=1 
num_output_channels=3
input_dim_1=28
input_dim_2=28
kernel_dim_1=5
kernel_dim_2=5
output_dim_1 = input_dim_1 - kernel_dim_1 + 1
output_dim_2 = input_dim_2 - kernel_dim_2 + 1

weights_init = GlorotUniformInit(weights_init_gain, rng)
biases_init = ConstantInit(biases_init)
error = CrossEntropySoftmaxError()
learning_rule =MomentumLearningRule(learning_rate,mom_coeff)
data_monitors = {'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}
weights_penalties = 0

rng.seed(seed)
train_data.reset()
valid_data.reset()

model = MultipleLayerModel([
             ReshapeLayer((num_input_channels,input_dim_1,input_dim_2)),
             ConvolutionalLayer(num_input_channels, num_output_channels,input_dim_1, input_dim_2,kernel_dim_1, kernel_dim_2), 
             ReluLayer(),
             ReshapeLayer((num_output_channels*output_dim_1*output_dim_1,)),     
             AffineLayer(3*24*24,hidden_dim,weights_init,biases_init),
             ReluLayer(),
             AffineLayer(hidden_dim,hidden_dim,weights_init,biases_init),
             ReluLayer(),
             AffineLayer(hidden_dim,output_dim,weights_init,biases_init)
         ])

optimiser = Optimiser(model, error, learning_rule, train_data, valid_data, data_monitors)
stats2,keys2,run_time2 = optimiser.train(num_epochs, stats_interval=stats_interval)


# In[30]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.style.use('ggplot')

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

for k in ['acc(train)',]:
    ax1.plot(np.arange(1, stats3.shape[0]) * stats_interval, 
            stats3[1:, keys3[k]], label='baseline')
    ax1.plot(np.arange(1, stats1.shape[0]) * stats_interval, 
                stats1[1:, key1[k]], label='withpooling')
    ax1.plot(np.arange(1, stats2.shape[0]) * stats_interval, 
            stats2[1:, keys2[k]], label='nopooling')

for k in ['acc(valid)',]:
    ax2.plot(np.arange(1, stats3.shape[0]) * stats_interval, 
            stats3[1:, keys3[k]], label='baseline')
    ax2.plot(np.arange(1, stats1.shape[0]) * stats_interval, 
                stats1[1:, key1[k]], label='withpooling')
    ax2.plot(np.arange(1, stats2.shape[0]) * stats_interval, 
            stats2[1:, keys2[k]], label='nopooling')
    
ax1.legend(loc=0)
ax1.set_xlabel('Epoch number')
ax1.set_ylabel('Training set accuracy')
ax2.legend(loc=0)
ax2.set_xlabel('Epoch number')
ax2.set_ylabel('Validation set accuracy')


# kernel_dim_1=1,kernel_dim_2=1

# In[31]:

# Seed a random number generator
seed = 24102016 
rng = np.random.RandomState(seed)

num_epochs = 50
stats_interval = 5
batch_size = 50
learning_rate = 0.001
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
num_input_channels=1 
num_output_channels=3
input_dim_1=28
input_dim_2=28
kernel_dim_1=1
kernel_dim_2=1
output_dim_1 = input_dim_1 - kernel_dim_1 + 1
output_dim_2 = input_dim_2 - kernel_dim_2 + 1

weights_init = GlorotUniformInit(weights_init_gain, rng)
biases_init = ConstantInit(biases_init)
error = CrossEntropySoftmaxError()
learning_rule =MomentumLearningRule(learning_rate,mom_coeff)
data_monitors = {'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}
weights_penalties = 0

rng.seed(seed)
train_data.reset()
valid_data.reset()

model = MultipleLayerModel([
             ReshapeLayer((num_input_channels,input_dim_1,input_dim_2)),
             ConvolutionalLayer(num_input_channels, num_output_channels,input_dim_1, input_dim_2,kernel_dim_1, kernel_dim_2), 
             ReluLayer(),
             ReshapeLayer((num_output_channels*output_dim_1*output_dim_1,)),
             MaxPoolingLayer(4),     
             AffineLayer(3*7*28,hidden_dim,weights_init,biases_init),
             ReluLayer(),
             AffineLayer(hidden_dim,hidden_dim,weights_init,biases_init),
             ReluLayer(),
             AffineLayer(hidden_dim,output_dim,weights_init,biases_init)
         ])

optimiser = Optimiser(model, error, learning_rule, train_data, valid_data, data_monitors)
stats1,key1,run_time1 = optimiser.train(num_epochs, stats_interval=stats_interval)


# kernel_dim_1=3, kernel_dim_2=3

# In[37]:

# Seed a random number generator
seed = 24102016 
rng = np.random.RandomState(seed)

num_epochs = 50
stats_interval = 5
batch_size = 50
learning_rate = 0.001
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
num_input_channels=1 
num_output_channels=3
input_dim_1=28
input_dim_2=28
kernel_dim_1=3
kernel_dim_2=3
output_dim_1 = input_dim_1 - kernel_dim_1 + 1
output_dim_2 = input_dim_2 - kernel_dim_2 + 1

weights_init = GlorotUniformInit(weights_init_gain, rng)
biases_init = ConstantInit(biases_init)
error = CrossEntropySoftmaxError()
learning_rule =MomentumLearningRule(learning_rate,mom_coeff)
data_monitors = {'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}
weights_penalties = 0

rng.seed(seed)
train_data.reset()
valid_data.reset()

model = MultipleLayerModel([
             ReshapeLayer((num_input_channels,input_dim_1,input_dim_2)),
             ConvolutionalLayer(num_input_channels, num_output_channels,input_dim_1, input_dim_2,kernel_dim_1, kernel_dim_2), 
             ReluLayer(),
             ReshapeLayer((num_output_channels*output_dim_1*output_dim_1,)),
             MaxPoolingLayer(2),     
             AffineLayer(3*13*26,hidden_dim,weights_init,biases_init),
             ReluLayer(),
             AffineLayer(hidden_dim,hidden_dim,weights_init,biases_init),
             ReluLayer(),
             AffineLayer(hidden_dim,output_dim,weights_init,biases_init)
         ])

optimiser = Optimiser(model, error, learning_rule, train_data, valid_data, data_monitors)
stats1,key1,run_time1 = optimiser.train(num_epochs, stats_interval=stats_interval)

