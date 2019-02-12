""" Wrapper functions for TensorFlow layers.
"""

import numpy as np
import tensorflow as tf
import graph_laplacian as graph


def _weight_variable(shape, regularization=True):
    initial = tf.truncated_normal_initializer(0, 0.1)
    var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
    tf.summary.histogram(var.op.name, var)

    if regularization:
        return var, tf.nn.l2_loss(var)
    return var


def _bias_variable(shape, regularization=True):
    initial = tf.constant_initializer(0.1)
    var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
    tf.summary.histogram(var.op.name, var)

    if regularization:
        return var, tf.nn.l2_loss(var)
    return var


def _variable_on_cpu(name, shape, initializer, use_fp16=False):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
      use_xavier: bool, whether to use xavier initializer

    Returns:
      Variable Tensor
    """
    if use_xavier:
        initializer = tf.contrib.layers.xavier_initializer()
    else:
        initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = _variable_on_cpu(name, shape, initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def conv1d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=1,
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
    """ 1D convolution with non-linear operation.

    Args:
      inputs: 3-D tensor variable BxLxC
      num_output_channels: int
      kernel_size: int
      scope: string
      stride: int
      padding: 'SAME' or 'VALID'
      use_xavier: bool, use xavier_initializer if true
      stddev: float, stddev for truncated_normal init
      weight_decay: float
      activation_fn: function
      bn: bool, whether to use batch norm
      bn_decay: float or float tensor variable in [0,1]
      is_training: bool Tensor variable

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_size,
                        num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        outputs = tf.nn.conv1d(inputs, kernel,
                               stride=stride,
                               padding=padding)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv1d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
    """ 2D convolution with non-linear operation.

    Args:
      inputs: 4-D tensor variable BxHxWxC
      num_output_channels: int
      kernel_size: a list of 2 ints
      scope: string
      stride: a list of 2 ints
      padding: 'SAME' or 'VALID'
      use_xavier: bool, use xavier_initializer if true
      stddev: float, stddev for truncated_normal init
      weight_decay: float
      activation_fn: function
      bn: bool, whether to use batch norm
      bn_decay: float or float tensor variable in [0,1]
      is_training: bool Tensor variable

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h, kernel_w,
                        num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        stride_h, stride_w = stride
        outputs = tf.nn.conv2d(inputs, kernel,
                               [1, stride_h, stride_w, 1],
                               padding=padding)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv2d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def conv2d_transpose(inputs,
                     num_output_channels,
                     kernel_size,
                     scope,
                     stride=[1, 1],
                     padding='SAME',
                     use_xavier=True,
                     stddev=1e-3,
                     weight_decay=0.0,
                     activation_fn=tf.nn.relu,
                     bn=False,
                     bn_decay=None,
                     is_training=None):
    """ 2D convolution transpose with non-linear operation.

    Args:
      inputs: 4-D tensor variable BxHxWxC
      num_output_channels: int
      kernel_size: a list of 2 ints
      scope: string
      stride: a list of 2 ints
      padding: 'SAME' or 'VALID'
      use_xavier: bool, use xavier_initializer if true
      stddev: float, stddev for truncated_normal init
      weight_decay: float
      activation_fn: function
      bn: bool, whether to use batch norm
      bn_decay: float or float tensor variable in [0,1]
      is_training: bool Tensor variable

    Returns:
      Variable tensor

    Note: conv2d(conv2d_transpose(a, num_out, ksize, stride), a.shape[-1], ksize, stride) == a
    """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h, kernel_w,
                        num_output_channels, num_in_channels]  # reversed to conv2d
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        stride_h, stride_w = stride

        # from slim.convolution2d_transpose
        def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
            dim_size *= stride_size

            if padding == 'VALID' and dim_size is not None:
                dim_size += max(kernel_size - stride_size, 0)
            return dim_size

        # calculate output shape
        batch_size = inputs.get_shape()[0].value
        height = inputs.get_shape()[1].value
        width = inputs.get_shape()[2].value
        out_height = get_deconv_dim(height, stride_h, kernel_h, padding)
        out_width = get_deconv_dim(width, stride_w, kernel_w, padding)
        output_shape = [batch_size, out_height, out_width, num_output_channels]

        outputs = tf.nn.conv2d_transpose(inputs, kernel, output_shape,
                                         [1, stride_h, stride_w, 1],
                                         padding=padding)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv2d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def conv3d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
    """ 3D convolution with non-linear operation.

    Args:
      inputs: 5-D tensor variable BxDxHxWxC
      num_output_channels: int
      kernel_size: a list of 3 ints
      scope: string
      stride: a list of 3 ints
      padding: 'SAME' or 'VALID'
      use_xavier: bool, use xavier_initializer if true
      stddev: float, stddev for truncated_normal init
      weight_decay: float
      activation_fn: function
      bn: bool, whether to use batch norm
      bn_decay: float or float tensor variable in [0,1]
      is_training: bool Tensor variable

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_d, kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_d, kernel_h, kernel_w,
                        num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        stride_d, stride_h, stride_w = stride
        outputs = tf.nn.conv3d(inputs, kernel,
                               [1, stride_d, stride_h, stride_w, 1],
                               padding=padding)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv3d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def SGC_LL(X,
           size_x,
           max_n_node,
           laplacian,
           num_output_channels,
           scope,
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           bn_decay=None,
           epsilon=1e-5,
           dropout=True,
           K=3,
           relu=True,
           is_training=None):
    """AGCN SGC_LL layer, with batch input of same graph node number (same graph size)
    X -> input batch of node features
    L -> input batch of intrinsic Laplacian matrix
    """
    with tf.variable_scope(scope) as sc:
        num_input_channels = X.get_shape()[-1].value
        weight = _variable_with_weight_decay('weights',
                                             shape=[num_input_channels * K, num_output_channels],
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        metric = _variable_with_weight_decay('metric',
                                             shape=[num_input_channels, num_input_channels],
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.constant_initializer(0.0))
        alpha = tf.get_variable('alpha', (1,), initializer=tf.constant_initializer(0.01), dtype=tf.float32)
        # gamma = tf.get_variable('gamma', (1,), initializer=tf.constant_initializer(1.0), dtype=tf.float32)

        def res_laplacian(X, W):
            transformed_X = tf.matmul(X, W)    # N * f_n, N = X.shape[0]
            sqr_X = tf.matmul(transformed_X, tf.transpose(transformed_X))   # N * N
            diag = tf.diag_part(sqr_X)  # N * 1
            shape_x = tf.slice(tf.shape(X), [0, ], [1, ])
            dup_diag = tf.reshape(tf.tile(diag, shape_x),
                                  tf.squeeze(tf.stack([shape_x, shape_x])))   # N * N
            l2_dist = tf.nn.l2_normalize(sqr_X + dup_diag + tf.transpose(dup_diag), dim=0, epsilon=epsilon)
            gaussian = alpha * tf.exp(tf.multiply(l2_dist, -1))

            # calculate Laplacian
            D = tf.reduce_sum(gaussian, axis=0)
            D = tf.diag(tf.squeeze(D))
            laplacian = D - gaussian
            return laplacian

        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin
            return tf.concat([x, x_], axis=0)  # K x M x Fin

        conved_x = []
        new_laps = []
        res_laps = []
        # batch_size, n_node = X.get_shape()[0].value, X.get_shape()[1].value
        batch_size = X.get_shape()[0].value
        list_x = tf.unstack(tf.squeeze(X), axis=0)
        list_lap = tf.unstack(tf.squeeze(laplacian), axis=0)
        list_size = tf.unstack(tf.squeeze(size_x), axis=0)
        for node_feature, orig_lap, n_node in zip(list_x, list_lap, list_size):

            node_feature = node_feature[: n_node]
            orig_lap = orig_lap[: n_node, : n_node]
            res_lap = res_laplacian(node_feature, metric)
            lap = orig_lap + res_lap

            x0 = node_feature
            x = tf.expand_dims(x0, 0)  # x-> 1 x M x Fin

            # Chebyshev recurrence T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)
            x1 = tf.matmul(lap, x0)
            x = concat(x, x1)
            for k in range(2, K):
                x2 = 2 * tf.matmul(lap, x1) - x0  # M x Fin
                x = concat(x, x2)
                x0, x1 = x1, x2

            shape = tf.stack([K, n_node, num_input_channels])
            shape2 = tf.stack([n_node, K * num_input_channels])
            # print('k-value:' + str(K))
            # print('x dim 0 ' + str(x.get_shape()[0].value))
            # print('x dim 1 ' + str(x.get_shape()[1].value))
            # print('x dim 2 ' + str(x.get_shape()[2].value))
            # debug = [x.get_shape()[1], x.get_shape()[2]]
            x = tf.reshape(x, shape)
            x = tf.transpose(x, perm=[1, 2, 0])  # x -> M x Fin x K
            x = tf.reshape(x, shape2)  # x-> M x (Fin*K)
            x = tf.matmul(x, weight)  # x -> M x Fout + Fout
            x = tf.nn.bias_add(x, biases)
            if relu:
                x = tf.nn.relu(x)
            x = tf.pad(x,
                       paddings=[[0, tf.constant(max_n_node) - n_node],
                                 [0, 0]],
                       mode="CONSTANT")
            conved_x.append(x)

            res_lap = tf.pad(res_lap,
                         paddings=[[0, tf.constant(max_n_node) - n_node],
                                   [0, tf.constant(max_n_node) - n_node]],
                         mode="CONSTANT")
            res_laps.append(res_lap)
            lap = tf.pad(lap,
                             paddings=[[0, tf.constant(max_n_node) - n_node],
                                       [0, tf.constant(max_n_node) - n_node]],
                             mode="CONSTANT")
            new_laps.append(lap)

        new_x = tf.stack(conved_x, axis=0)
        new_x = tf.reshape(new_x, [batch_size, max_n_node, 1, num_output_channels])

        new_L = tf.stack(new_laps, axis=0)
        new_L = tf.reshape(new_L, [batch_size, max_n_node, max_n_node])

        res_L = tf.stack(res_laps, axis=0)
        res_L = tf.reshape(res_L, [batch_size, max_n_node, max_n_node])

        # batch normalization
        if bn_decay is not None:
            new_x = batch_norm_for_conv1d(new_x, is_training, bn_decay, scope)
        if dropout:
            new_x = tf.nn.dropout(new_x, keep_prob=0.8)
        return new_x, new_L, res_L


def SGC_LL2(X,
            num_output_channels,
            scope,
            use_xavier=True,
            stddev=1e-3,
            weight_decay=0.0,
            bn_decay=None,
            epsilon=1e-5,
            dropout=True,
            K=2,
            is_training=None):
    """AGCN SGC_LL layer, with batch input of same graph node number (same graph size)
    X -> input batch of node features
    L -> input batch of intrinsic Laplacian matrix
    NO original graph Laplacian as inputs, one-sample Input, batch_size == 1
    """
    with tf.variable_scope(scope) as sc:
        num_input_channels = X.get_shape()[-1].value
        n_node = tf.shape(X)[0]
        weight = _variable_with_weight_decay('weights',
                                             shape=[num_input_channels * K, num_output_channels],
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        metric = _variable_with_weight_decay('metric',
                                             shape=[num_input_channels, num_input_channels],
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.constant_initializer(0.0))
        alpha = tf.get_variable('alpha', (1,), initializer=tf.constant_initializer(0.01), dtype=tf.float32)
        # gamma = tf.get_variable('gamma', (1,), initializer=tf.constant_initializer(1.0), dtype=tf.float32)

        def res_laplacian(X, W):
            transformed_X = tf.matmul(X, W)    # N * f_n, N = X.shape[0]
            sqr_X = tf.matmul(transformed_X, tf.transpose(transformed_X))   # N * N
            diag = tf.diag_part(sqr_X)  # N * 1
            shape_x = tf.slice(tf.shape(X), [0, ], [1, ])
            dup_diag = tf.reshape(tf.tile(diag, shape_x),
                                  tf.squeeze(tf.stack([shape_x, shape_x])))   # N * N
            l2_dist = tf.nn.l2_normalize(sqr_X + dup_diag + tf.transpose(dup_diag), dim=0, epsilon=epsilon)
            gaussian = alpha * tf.exp(tf.multiply(l2_dist, -1))

            # calculate Laplacian
            D = tf.reduce_sum(gaussian, axis=0)
            D = tf.diag(tf.squeeze(D))
            laplacian = D - gaussian
            return laplacian

        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin
            return tf.concat([x, x_], axis=0)  # K x M x Fin


        # batch_size = tf.shape(X)[0]
        # X = tf.squeeze(X, axis=0)
        # list_x = tf.unstack(X, axis=0)
        # list_size = tf.unstack(tf.squeeze(size_x), axis=0)
        # for node_feature in list_x:

        lap = res_laplacian(X, metric)

        x0 = X
        x = tf.expand_dims(x0, 0)  # x-> 1 x M x Fin

        # Chebyshev recurrence T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)
        x1 = tf.matmul(lap, x0)
        x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * tf.matmul(lap, x1) - x0  # M x Fin
            x = concat(x, x2)
            x0, x1 = x1, x2

        shape = tf.stack([K, n_node, num_input_channels])
        shape2 = tf.stack([n_node, K * num_input_channels])

        x = tf.reshape(x, shape)
        x = tf.transpose(x, perm=[1, 2, 0])  # x -> M x Fin x K
        x = tf.reshape(x, shape2)  # x-> M x (Fin*K)
        x = tf.matmul(x, weight)  # x -> M x Fout + Fout
        x = tf.nn.bias_add(x, biases)
        x = tf.nn.relu(x)

        # new_x = tf.stack(x, axis=0)
        new_x = tf.reshape(x, [n_node, num_output_channels])

        # batch normalization
        if bn_decay is not None:
            new_x = batch_norm_for_conv1d(new_x, is_training, bn_decay, scope)
        if dropout:
            new_x = tf.nn.dropout(new_x, keep_prob=0.8)
        return new_x


def GCN_chebyshev(X,
                  size_x,
                  max_n_node,
                  laplacian,
                  num_output_channels,
                  scope,
                  K=3,
                  dropout=True,
                  relu=False,
                  bn_decay=None,
                  is_training=True,):
    """ Michaël Defferrard, Xavier Bresson, Pierre Vandergheynst,
        Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering, NIPS, 2016.
    """

    with tf.variable_scope(scope) as sc:
        num_input_channels = X.get_shape()[-1].value  # N: number of samples, M: number of features

        "initialize weights bias"
        weight, reg_loss_W = _weight_variable([K * num_input_channels, num_output_channels])
        bias, reg_loss_b = _bias_variable([num_output_channels])

        "list of regularizations terms on weights"
        regs = [reg_loss_b, reg_loss_W]

        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin
            return tf.concat([x, x_], axis=0)  # K x M x Fin

        conved_x = []
        # batch_size, n_node = X.get_shape()[0].value, X.get_shape()[1].value
        batch_size = X.get_shape()[0].value
        list_x = tf.unstack(tf.squeeze(X), axis=0)
        list_lap = tf.unstack(tf.squeeze(laplacian), axis=0)
        list_size = tf.unstack(tf.squeeze(size_x), axis=0)
        for node_feature, orig_lap, n_node in zip(list_x, list_lap, list_size):

            node_feature = node_feature[: n_node]
            orig_lap = orig_lap[: n_node, : n_node]
            lap = orig_lap

            x0 = node_feature
            x = tf.expand_dims(x0, 0)  # x-> 1 x M x Fin

            # Chebyshev recurrence T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)
            x1 = tf.matmul(lap, x0)
            x = concat(x, x1)
            for k in range(2, K):
                x2 = 2 * tf.matmul(lap, x1) - x0  # M x Fin
                x = concat(x, x2)
                x0, x1 = x1, x2

            shape = tf.stack([K, n_node, num_input_channels])
            shape2 = tf.stack([n_node, K * num_input_channels])
            # print('k-value:' + str(K))
            # print('x dim 0 ' + str(x.get_shape()[0].value))
            # print('x dim 1 ' + str(x.get_shape()[1].value))
            # print('x dim 2 ' + str(x.get_shape()[2].value))
            # debug = [x.get_shape()[1], x.get_shape()[2]]
            x = tf.reshape(x, shape)
            x = tf.transpose(x, perm=[1, 2, 0])  # x -> M x Fin x K
            x = tf.reshape(x, shape2)  # x-> M x (Fin*K)
            x = tf.matmul(x, weight)  # x -> M x Fout + Fout
            x = tf.nn.bias_add(x, bias)
            if relu:
                x = tf.nn.relu(x)
            x = tf.pad(x,
                       paddings=[[0, tf.constant(max_n_node) - n_node],
                                 [0, 0]],
                       mode="CONSTANT")
            conved_x.append(x)

        new_x = tf.stack(conved_x, axis=0)
        new_x = tf.reshape(new_x, [batch_size, max_n_node, 1, num_output_channels])

        # batch normalization
        if bn_decay is not None:
            new_x = batch_norm_for_conv1d(new_x, is_training, bn_decay, scope)
        if dropout:
            new_x = tf.nn.dropout(new_x, keep_prob=0.8)
    return new_x, regs


def GCN_lanczos(X, laplacian, size_x,  num_output_channels, K, max_n_node, scope):
    with tf.variable_scope(scope) as sc:
        "initialize weights"
        weight, reg_loss_W = _weight_variable([K, num_output_channels])

        "list of regularizations terms on weights"
        regs = [reg_loss_W]

        batch_size = X.get_shape()[0].value
        list_x = tf.unstack(tf.squeeze(X), axis=0)
        list_lap = tf.unstack(tf.squeeze(laplacian), axis=0)
        list_size = tf.unstack(tf.squeeze(size_x), axis=0)

        conved_x = []
        for node_feature, orig_lap, n_node in zip(list_x, list_lap, list_size):

            node_feature = node_feature[: n_node]
            M, num_input_channels = tf.squeeze(node_feature).get_shape()
            orig_lap = orig_lap[: n_node, : n_node]
            L = graph.rescale_L(orig_lap, lmax=2)  # Graph Laplacian, M x M

            x = node_feature
            xl = tf.transpose(x)  # num_input_channels x M

            def lanczos(x):
                return graph.lanczos(L, x, K)

            xl = tf.py_func(lanczos, [xl], [tf.float32])[0]
            xl = tf.transpose(xl)  # N x M x K
            xl = tf.reshape(xl, [-1, K])  # NM x K
            # Filter
            xt = tf.matmul(xl, weight)  # NM x F
            xt = tf.reshape(xt, [-1, M, num_output_channels])  # N x M x F
            xt = tf.nn.relu(xt)
            x = tf.pad(xt,
                       paddings=[[0, tf.constant(max_n_node) - n_node],
                                 [0, 0]],
                       mode="CONSTANT")
            conved_x.append(x)

    new_x = tf.stack(conved_x, axis=0)
    new_x = tf.reshape(new_x, [batch_size, max_n_node, 1, num_output_channels])

    return new_x, regs


def gather(X, size_x, attentions=None, feature=False):
    num_output_channels = X.get_shape()[-1].value
    batch_size = X.get_shape()[0].value
    list_x = tf.unstack(tf.squeeze(X, axis=2), axis=0)
    list_size = tf.unstack(tf.squeeze(size_x), axis=0)
    if attentions is not None:
        # multiple attention
        list_attn = tf.unstack(attentions, axis=0)

    output_x = []
    if attentions is not None:
        for node_feature, n_node, attn in zip(list_x, list_size, list_attn):
            node_feature = node_feature[: n_node]   # n x F
            node_feature = tf.transpose(node_feature, perm=(1, 0))  # F x n

            attn = tf.transpose(attn, perm=(1, 0))  # N X 3
            attn = attn[: n_node]   # n x 3

            attn_node_features = tf.matmul(node_feature, attn)  # num_f x 3 or 1 x 3

            if feature:
                # aggregate feature, flatten to 1-D array
                agg_node = tf.reshape(attn_node_features, shape=[-1])
            else:
                # aggregate prediction, get mean
                agg_node = tf.squeeze(tf.reduce_mean(attn_node_features, axis=1))

            output_x.append(agg_node)
        output_x = tf.stack(output_x, axis=0)
    else:
        for node_feature, n_node in zip(list_x, list_size):
            node_feature = node_feature[: n_node]
            sumed_x = tf.reduce_mean(node_feature, axis=0)
            output_x.append(sumed_x)
        output_x = tf.stack(output_x, axis=0)
        output_x = tf.squeeze(tf.reshape(output_x, [batch_size, num_output_channels]))

    return output_x


def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=0.0,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    is_training=None):
    """ Fully connected layer with non-linear operation.

    Args:
      inputs: 2-D tensor BxN
      num_outputs: int

    Returns:
      Variable tensor of size B x num_outputs.
    """
    with tf.variable_scope(scope) as sc:
        num_input_units = inputs.get_shape()[-1].value
        weights = _variable_with_weight_decay('weights',
                                              shape=[num_input_units, num_outputs],
                                              use_xavier=use_xavier,
                                              stddev=stddev,
                                              wd=weight_decay)
        outputs = tf.matmul(inputs, weights)
        biases = _variable_on_cpu('biases', [num_outputs],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_fc(outputs, is_training, bn_decay, 'bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def max_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2,2],
               padding='VALID',
               use_relu=False):
    """ 2D max pooling.

    Args:
      inputs: 4-D tensor BxHxWxC
      kernel_size: a list of 2 ints
      stride: a list of 2 ints

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf.nn.max_pool(inputs,
                                 ksize=[1, kernel_h, kernel_w, 1],
                                 strides=[1, stride_h, stride_w, 1],
                                 padding=padding,
                                 name=sc.name)
        if use_relu:
            outputs = tf.nn.relu(outputs)
        return outputs


def avg_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
    """ 2D avg pooling.

    Args:
      inputs: 4-D tensor BxHxWxC
      kernel_size: a list of 2 ints
      stride: a list of 2 ints

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf.nn.avg_pool(inputs,
                                 ksize=[1, kernel_h, kernel_w, 1],
                                 strides=[1, stride_h, stride_w, 1],
                                 padding=padding,
                                 name=sc.name)
        return outputs


def max_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
    """ 3D max pooling.

    Args:
      inputs: 5-D tensor BxDxHxWxC
      kernel_size: a list of 3 ints
      stride: a list of 3 ints

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_d, kernel_h, kernel_w = kernel_size
        stride_d, stride_h, stride_w = stride
        outputs = tf.nn.max_pool3d(inputs,
                                   ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                                   strides=[1, stride_d, stride_h, stride_w, 1],
                                   padding=padding,
                                   name=sc.name)
        return outputs


def avg_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
    """ 3D avg pooling.

    Args:
      inputs: 5-D tensor BxDxHxWxC
      kernel_size: a list of 3 ints
      stride: a list of 3 ints

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_d, kernel_h, kernel_w = kernel_size
        stride_d, stride_h, stride_w = stride
        outputs = tf.nn.avg_pool3d(inputs,
                                   ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                                   strides=[1, stride_d, stride_h, stride_w, 1],
                                   padding=padding,
                                   name=sc.name)
        return outputs


def batch_norm_template(inputs, is_training, scope, moments_dims, bn_decay):
    """ Batch normalization on convolutional maps and beyond...
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow

    Args:
        inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
        is_training:   boolean tf.Varialbe, true indicates training phase
        scope:         string, variable scope
        moments_dims:  a list of ints, indicating dimensions for moments calculation
        bn_decay:      float or float tensor variable, controling moving average weight
    Return:
        normed:        batch-normalized maps
    """
    with tf.variable_scope(scope) as sc:
        num_channels = inputs.get_shape()[-1].value
        beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
        decay = bn_decay if bn_decay is not None else 0.9
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        # Operator that maintains moving averages of variables.
        ema_apply_op = tf.cond(is_training,
                               lambda: ema.apply([batch_mean, batch_var]),
                               lambda: tf.no_op())

        # Update moving average and return current batch's avg and var.
        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # ema.average returns the Variable holding the average of var.
        mean, var = tf.cond(is_training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
    return normed


def batch_norm_for_fc(inputs, is_training, bn_decay, scope):
    """ Batch normalization on FC data.

    Args:
        inputs:      Tensor, 2D BxC input
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0, ], bn_decay)


def batch_norm_for_conv1d(inputs, is_training, bn_decay, scope):
    """ Batch normalization on 1D convolutional maps.

    Args:
        inputs:      Tensor, 3D BLC input maps
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0, 1], bn_decay)


def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope):
    """ Batch normalization on 2D convolutional maps.

    Args:
        inputs:      Tensor, 4D BHWC input maps
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0, 1, 2], bn_decay)


def batch_norm_for_conv3d(inputs, is_training, bn_decay, scope):
    """ Batch normalization on 3D convolutional maps.

    Args:
        inputs:      Tensor, 5D BDHWC input maps
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0, 1, 2, 3], bn_decay)


def dropout(inputs,
            is_training,
            scope,
            keep_prob=0.5,
            noise_shape=None):
    """ Dropout layer.

    Args:
      inputs: tensor
      is_training: boolean tf.Variable
      scope: string
      keep_prob: float in [0,1]
      noise_shape: list of ints

    Returns:
      tensor variable
    """
    with tf.variable_scope(scope) as sc:
        outputs = tf.cond(is_training,
                          lambda: tf.nn.dropout(inputs, keep_prob, noise_shape),
                          lambda: inputs)
        return outputs


def flatten(inputs):
    """
    Handy function for flattening the result of a conv2D or
    maxpool2D to be used for a fully-connected (affine) layer.
    """
    layer_shape = inputs.get_shape()
    # num_features = tf.reduce_prod(tf.shape(layer)[1:])
    num_features = tf.cast(layer_shape[1:].num_elements(), dtype=tf.int32)
    layer_flat = tf.reshape(inputs, [-1, num_features])

    return layer_flat
