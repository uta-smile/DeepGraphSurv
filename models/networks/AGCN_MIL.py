import tensorflow as tf
import numpy as np
import math
import sys
import os

BASE_DIR = os.path.join(os.environ["HOME"], 'AGCN_tf/AGCN_tf')
sys.path.append(os.path.join(BASE_DIR, 'models/networks'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import layers


def placeholder_inputs(batch_size, n_patch, height, width):
    image_pl = tf.placeholder(tf.float32, shape=(None, height, width, 3))
    labels_pl = tf.placeholder(tf.float32, shape=(1,))

    return image_pl, labels_pl


def attention_net(input, is_training, bn_decay=None):
    """ Attention Net, input is BxNxF
        Return:
            Attention matrix, BxN """
    """ TF accept variable shape by tf.shape()[], if use *.get_shape()[], it must be specified"""
    # batch_size = tf.shape(input)[0]
    # n_node = tf.shape(input)[0]
    F_num = input.get_shape()[-1].value
    input_trans = tf.reshape(input, [-1, F_num])    # (B x n_node) x F_out

    with tf.variable_scope('attention_net', reuse=tf.AUTO_REUSE) as attn_net:

        net = layers.fully_connected(input_trans, 128,  weight_decay=0.0005,
                                     bn=True, is_training=is_training, activation_fn=tf.nn.tanh,
                                     scope='fc1', bn_decay=bn_decay)
        net = layers.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')

        net = layers.fully_connected(net, 1,  weight_decay=0.0005, bn=True, is_training=is_training,
                                     activation_fn=None, scope='fc4', bn_decay=bn_decay)
        attention = tf.nn.softmax(net, dim=0)

    return attention


def featurizer_cnn(image, is_training, bn_decay=None):
    """
    extract features from image (cell small images 5-D Tensor, B x N x H x W x 3)
    B -> batch size, number of image (patch)
    N -> number of small image (as graph node) from ONE image, we first fix it
    """
    n_cell = tf.shape(image)[0]

    with tf.variable_scope('feature_net', reuse=tf.AUTO_REUSE) as f_net:

        net = layers.conv2d(image, 36, [4, 4],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training,
                            weight_decay=0.0005,
                            scope='conv1', bn_decay=bn_decay)
        net = layers.max_pool2d(net, [2,2], padding='VALID', scope='mappool1')
        net = layers.conv2d(net, 48, [3, 3],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training,
                            weight_decay=0.0005,
                            scope='conv2', bn_decay=bn_decay)
        net = layers.max_pool2d(net, [2,2], padding='VALID', scope='mappool2')

        res_1 = net.get_shape()[1].value
        res_2 = net.get_shape()[2].value
        f_out = net.get_shape()[3].value

        net = tf.reshape(net, [n_cell, res_1 * res_2 * f_out])       # B x N x F_out

        net = layers.fully_connected(net, 512, weight_decay=0.0005, bn=True,
                                     is_training=is_training, scope='fc1', bn_decay=bn_decay)
        net = layers.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
        net = layers.fully_connected(net, 512, weight_decay=0.0005, bn=True,
                                     is_training=is_training, scope='fc2', bn_decay=bn_decay)
        cell_features = layers.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp2')

    return cell_features


def agcnMIL_wsi_cls(image, n_classes, is_training, bn_decay=None):
    """ Classification AGCN + MIL + Attention + PointNet, input is BxNxHxWx3, output Bx2 """
    # batch_size = image.get_shape()[0].value  # number of patch in a WSI
    # n_node = image.get_shape()[1].value

    if is_training is True:
        n_cell = tf.to_float(tf.shape(image)[0])
        augment_ratio = tf.constant(0.3, dtype=tf.float32)
        sel_n_cell = tf.to_int32(tf.multiply(n_cell, augment_ratio))
        # idx = np.arange(n_cell)
        # np.random.shuffle(idx)
        # img_shff = image[idx]
        angles = tf.random_uniform((sel_n_cell,), maxval=180, seed=1)
        rotate_sel_img = tf.contrib.image.rotate(image[:sel_n_cell], angles)
        image = tf.concat([image, rotate_sel_img], axis=0)

    end_points = dict()
    data = featurizer_cnn(image, is_training, bn_decay=bn_decay)    # Bx N x F
    attn = attention_net(data, is_training, bn_decay=bn_decay)      # B x N ~(0,1)
    end_points['attn'] = tf.squeeze(attn)
    with tf.variable_scope('agcn_mil', reuse=tf.AUTO_REUSE) as agcn_mil:

        net1 = layers.SGC_LL2(
            data,
            num_output_channels=128,
            scope='sgc_ll_1',
            is_training=is_training,
            bn_decay=bn_decay)

        # net1 = layers.SGC_LL2(
        #     net1,
        #     num_output_channels=256,
        #     scope='sgc_ll_3',
        #     is_training=is_training,
        #     bn_decay=bn_decay)

        net2 = layers.SGC_LL2(
            net1,
            num_output_channels=1024,
            scope='sgc_ll_2',
            is_training=is_training,
            bn_decay=bn_decay)
        #
        graphs = net2   # BxNxFout
        graphs_trans = tf.transpose(graphs, perm=(1, 0))     # Fout x N
        attned_graph = tf.matmul(graphs_trans, attn)    # Fout x (B)

        # attned_graph = tf.matmul(tf.transpose(data, (1, 0)), attn)

        output1 = tf.squeeze(attned_graph, axis=-1)
        # end_points['graph_representation'] = output1

    with tf.variable_scope('classification', reuse=tf.AUTO_REUSE) as cls:

        net = tf.expand_dims(output1, dim=0)
        # net = layers.fully_connected(output1, 64, bn=True, is_training=is_training,
        #                              scope='fc1', bn_decay=bn_decay, weight_decay=0.0005)
        # net = layers.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
        output2 = layers.fully_connected(net, 1, activation_fn=tf.nn.tanh, weight_decay=0.0005,
                                         is_training=is_training, scope='fc2')
        end_points['ss'] = output2
        output2 = tf.nn.sigmoid(tf.squeeze(output2, axis=-1))
        end_points['class_prediction'] = output2

    to_save_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "agcn_mil/") +\
                   tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "classification/")
    saver_op = tf.train.Saver({v.op.name: v for v in to_save_vars})
    end_points['saver'] = saver_op
    end_points['saved_vars'] = to_save_vars

    weight_decay = tf.get_collection('losses')
    end_points['weight_decay'] = tf.reduce_sum(weight_decay)
    return end_points


def agcn_loss(pred, regs, label, reg_weight=0.1):
    """ pred: B*NUM_CLASSES,
        label: B, """
    one_hot_label = tf.one_hot(label, depth=2)
    classify_loss = tf.losses.softmax_cross_entropy(logits=pred, onehot_labels=one_hot_label)
    # classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)

    regs = tf.reduce_mean(regs)
    tf.summary.scalar('Laplacian l2 loss', regs)

    return classify_loss + reg_weight * regs


def cross_entropy(end_points, label):
    pred = end_points['class_prediction']
    weight_loss = end_points['weight_decay']
    # label = tf.one_hot(label, depth=2)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_sum(loss)
    tf.summary.scalar('classify loss', classify_loss)

    return classify_loss + weight_loss


if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((100, 27, 27, 3))
        outputs = agcnMIL_wsi_cls(inputs, 2, tf.constant(True))
        print(outputs)
