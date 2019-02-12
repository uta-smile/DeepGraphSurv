import tensorflow as tf
import numpy as np
import math
import sys
import os

BASE_DIR = os.path.join(os.environ["HOME"], 'AGCN_tf/AGCN_tf')
sys.path.append(os.path.join(BASE_DIR, 'models/networks'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import layers


def placeholder_inputs(batch_size, num_point, n_feature):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, n_feature))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, ))
    laplacian_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_point))
    size_x_pl = tf.placeholder(tf.int32, shape=(batch_size, ))

    return pointclouds_pl, labels_pl, laplacian_pl, size_x_pl


def placeholder_inputs_survival(batch_size, num_point, n_feature):
    # shape is flexible, maybe changed during training
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, n_feature))
    laplacian_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_point))
    size_x_pl = tf.placeholder(tf.int32, shape=(batch_size,))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size,))

    return pointclouds_pl, labels_pl, laplacian_pl, size_x_pl


def placeholder_survival(batch_size):
    # shape is not specified
    status_pl = tf.placeholder(tf.float32,shape=(batch_size,))
    return status_pl


def attention_net(input, laplacian, size_x, max_node_n, is_training, bn_decay=None):
    """ Attention Net, input is BxNxF
        Return:
            Attention matrix, BxN """
    """ TF accept variable shape by tf.shape()[], if use *.get_shape()[], it must be specified"""
    # batch_size = tf.shape(input)[0]
    # n_node = input.get_shape()[1].value
    # F_num = input.get_shape()[-1].value
    # input_trans = tf.reshape(input, [-1, n_node, F_num])    # (B x n_node) x F_out
    # input_trans = tf.reshape(input_trans, [-1, n_node * F_num])

    with tf.variable_scope('attention_net', reuse=tf.AUTO_REUSE) as attn_net:

        net1, _, _ = layers.SGC_LL(
            input,
            size_x,
            max_node_n,
            laplacian,
            num_output_channels=32,
            scope='sgc_ll_1',
            is_training=is_training,
            dropout=False,
            relu=False,
            bn_decay=bn_decay)
        net2, _, _ = layers.SGC_LL(
            net1,
            size_x,
            max_node_n,
            laplacian,
            num_output_channels=1,
            scope='sgc_ll_2',
            is_training=is_training,
            dropout=False,
            relu=False,
            bn_decay=bn_decay)
        net2 = net2 / 1e3
        net3 = tf.sigmoid(net2)    # sigmoid
        net3 = tf.squeeze(net3, axis=-1)
        attention = tf.nn.softmax(net3, dim=1)     # should be bxN N is the NO. of nodes
        attention = tf.transpose(attention, perm=(0, 2, 1))
    return attention, net3


def attention_net_fc(input, laplacian, size_x, max_node_n, is_training, bn_decay=None):
    """ Attention Net, input is BxNxF
        Return:
            Attention matrix, BxN """
    """ TF accept variable shape by tf.shape()[], if use *.get_shape()[], it must be specified"""
    # batch_size = tf.shape(input)[0]
    # n_node = input.get_shape()[1].value
    # F_num = input.get_shape()[-1].value
    # input_trans = tf.reshape(input, [-1, n_node, F_num])    # (B x n_node) x F_out
    # input_trans = tf.reshape(input_trans, [-1, n_node * F_num])

    with tf.variable_scope('attention_net', reuse=tf.AUTO_REUSE) as attn_net:

        net1, _, _ = layers.SGC_LL(
            input,
            size_x,
            max_node_n,
            laplacian,
            num_output_channels=32,
            scope='sgc_ll_1',
            is_training=is_training,
            bn_decay=bn_decay)
        net2, _, _ = layers.SGC_LL(
            net1,
            size_x,
            max_node_n,
            laplacian,
            num_output_channels=1,
            scope='sgc_ll_2',
            is_training=is_training,
            bn_decay=bn_decay)
        net2 = tf.squeeze(net2, axis=-1)
        attention = tf.nn.softmax(net2, dim=-1)     # should be bxN N is the NO. of nodes
    return attention


def multi_attention_net(input, laplacian, size_x, max_node_n, is_training, bn_decay=None):
    """ Attention Net, input is BxNxF
        Return:
            Attention matrix, BxN """
    """ TF accept variable shape by tf.shape()[], if use *.get_shape()[], it must be specified"""
    # batch_size = tf.shape(input)[0]
    # n_node = input.get_shape()[1].value
    # F_num = input.get_shape()[-1].value
    # input_trans = tf.reshape(input, [-1, n_node, F_num])    # (B x n_node) x F_out
    # input_trans = tf.reshape(input_trans, [-1, n_node * F_num])

    with tf.variable_scope('attention_net', reuse=tf.AUTO_REUSE) as attn_net:

        net1, _, _ = layers.SGC_LL(
            input,
            size_x,
            max_node_n,
            laplacian,
            num_output_channels=32,
            scope='sgc_ll_1',
            is_training=is_training,
            bn_decay=bn_decay)
        net21, _, _ = layers.SGC_LL(
            net1,
            size_x,
            max_node_n,
            laplacian,
            num_output_channels=1,
            scope='sgc_ll_21',
            is_training=is_training,
            bn_decay=bn_decay)
        net21 = tf.squeeze(net21)

        net22, _, _ = layers.SGC_LL(
            net1,
            size_x,
            max_node_n,
            laplacian,
            num_output_channels=1,
            scope='sgc_ll_22',
            is_training=is_training,
            bn_decay=bn_decay)
        net22 = tf.squeeze(net22)
        net23, _, _ = layers.SGC_LL(
            net1,
            size_x,
            max_node_n,
            laplacian,
            num_output_channels=1,
            scope='sgc_ll_23',
            is_training=is_training,
            bn_decay=bn_decay)
        net23 = tf.squeeze(net23)

        attention1 = tf.nn.softmax(net21, dim=-1)     # should be bxN N is the NO. of nodes
        attention2 = tf.nn.softmax(net22, dim=-1)     # should be bxN N is the NO. of nodes
        attention3 = tf.nn.softmax(net23, dim=-1)     # should be bxN N is the NO. of nodes

        all_attn = tf.stack([attention1, attention2, attention3])
        all_attn = tf.transpose(all_attn, perm=(1, 0, 2))
    return all_attn


def basic_agcn_wsi_survival(data,
                            laplacian,
                            size_x,
                            max_node_n,
                            n_classes,
                            is_training,
                            use_attn=True,
                            bn_decay=None):
    """ Classification AGCN + PointNet, input is BxNx3, output Bx40 """
    # batch_size = data.get_shape()[0].value
    # num_point = data.get_shape()[1].value
    end_points = dict()
    if use_attn:
        # B X K X N
        attn, an_out = attention_net(data, laplacian, size_x, max_node_n, is_training, bn_decay=None)
    else:
        attn = None
    end_points['attn'] = attn
    end_points['an_out'] = an_out

    with tf.variable_scope('backbone_net', reuse=tf.AUTO_REUSE) as agcn:

        net1, lap1, res_lap1 = layers.SGC_LL(
            data,
            size_x,
            max_node_n,
            laplacian,
            num_output_channels=256,
            scope='sgc_ll_1',
            is_training=is_training,
            bn_decay=bn_decay)

        net2, lap2, res_lap2 = layers.SGC_LL(
            net1,
            size_x,
            max_node_n,
            laplacian,
            num_output_channels=128,
            scope='sgc_ll_2',
            is_training=is_training,
            bn_decay=bn_decay)

        net3, lap3, res_lap3 = layers.SGC_LL(
            net2,
            size_x,
            max_node_n,
            laplacian,
            num_output_channels=n_classes,
            scope='sgc_ll_3',
            is_training=is_training,
            bn_decay=bn_decay)

        output = layers.gather(
            net3,
            size_x,
            attentions=attn,
        )
        end_points['regs_list'] = [tf.nn.l2_loss(l) for l in [res_lap1, res_lap3, res_lap2]]
        end_points['graph_feature'] = tf.squeeze(tf.reduce_mean(net2, axis=1))
        end_points['output'] = output
        end_points['node_pred'] = tf.squeeze(net3)
        to_save_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "backbone_net/")
        saver_op = tf.train.Saver({v.op.name: v for v in to_save_vars})
        end_points['saver'] = saver_op
        end_points['saved_vars'] = to_save_vars
        end_points['res_graph'] = res_lap3

    return end_points


def basic_agcn_wsi_survival2(data,
                            laplacian,
                            size_x,
                            max_node_n,
                            n_classes,
                            is_training,
                            use_attn=True,
                            bn_decay=None):
    """ Classification AGCN + PointNet, input is BxNx3, output Bx40 """
    # batch_size = data.get_shape()[0].value
    # num_point = data.get_shape()[1].value
    end_points = dict()
    if use_attn:
        attn, an_out = attention_net(data, laplacian, size_x, max_node_n, is_training, bn_decay=None)
    else:
        attn = None
    end_points['attn'] = attn
    end_points['an_out'] = an_out

    with tf.variable_scope('backbone_net', reuse=tf.AUTO_REUSE) as agcn:

        net1, lap1, res_lap1 = layers.SGC_LL(
            data,
            size_x,
            max_node_n,
            laplacian,
            num_output_channels=256,
            scope='sgc_ll_1',
            is_training=is_training,
            bn_decay=bn_decay)

        net2, lap2, res_lap2 = layers.SGC_LL(
            net1,
            size_x,
            max_node_n,
            laplacian,
            num_output_channels=256,
            scope='sgc_ll_2',
            is_training=is_training,
            bn_decay=bn_decay)

        net3, lap3, res_lap3 = layers.SGC_LL(
            net2,
            size_x,
            max_node_n,
            laplacian,
            num_output_channels=128,
            scope='sgc_ll_3',
            is_training=is_training,
            bn_decay=bn_decay)

        net4 = layers.gather(
            net3,
            size_x,
            attentions=attn,
            feature=True,
        )

        output = layers.fully_connected(net4, 64, activation_fn=tf.nn.tanh, weight_decay=0.0005,
                                        is_training=is_training, scope='fc1')
        output = layers.fully_connected(output, n_classes, activation_fn=tf.nn.tanh, weight_decay=0.0005,
                                        is_training=is_training, scope='fc2')
        end_points['regs_list'] = [tf.nn.l2_loss(l) for l in [res_lap1, res_lap3, res_lap2]]
        end_points['graph_feature'] = tf.squeeze(net4)
        end_points['output'] = output
        # Symmetric function: max pooling
        # net5 = layers.max_pool2d(net4, [num_point, 1],
        #                          padding='SAME', scope='maxpool')

        # output = tf.reduce_mean(tf.squeeze(net5), axis=1)    # sum all nodes of a graph
        # output_softmax = tf.nn.softmax(net5)
        # end_points['output_softmax'] = output_softmax

        to_save_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "backbone_net/")
        saver_op = tf.train.Saver({v.op.name: v for v in to_save_vars})
        end_points['saver'] = saver_op
        end_points['saved_vars'] = to_save_vars
        end_points['res_graph'] = res_lap3

    return end_points


def survival_loss(preds, y):
    # from Yao
    out = tf.reshape(tf.squeeze(preds), [-1])
    hazard_ratio = tf.exp(out)
    log_risk = tf.log(tf.cumsum(hazard_ratio))
    uncensored_likelihood = out - log_risk
    # y- status,
    censored_likelihood = tf.multiply(uncensored_likelihood, y)
    neg_log_loss = -tf.reduce_sum(censored_likelihood)

    return neg_log_loss

#
# def basic_agcn_smile(data, laplacian, size_x, max_node_n, n_classes, is_training, bn_decay=None):
#     """ Classification AGCN + PointNet, input is BxNx3, output Bx40
#         n_classes, output dimension, for regression, it could be 1
#     """
#     # batch_size = data.get_shape()[0].value
#     num_point = data.get_shape()[1].value
#     end_points = dict()
#
#     with tf.variable_scope('basic_agcn', reuse=tf.AUTO_REUSE) as agcn:
#         net1, lap1, res_lap1 = layers.SGC_LL(
#             data,
#             size_x,
#             max_node_n,
#             laplacian,
#             num_output_channels=256,
#             scope='sgc_ll_1',
#             is_training=is_training,
#             bn_decay=bn_decay)
#         net3, lap3, res_lap3 = layers.SGC_LL(
#             net1,
#             size_x,
#             max_node_n,
#             laplacian,
#             num_output_channels=32,
#             scope='sgc_ll_3',
#             is_training=is_training,
#             bn_decay=bn_decay)
#
#         net4, lap4, res_lap4 = layers.SGC_LL(
#             net3,
#             size_x,
#             max_node_n,
#             laplacian,
#             num_output_channels=n_classes,
#             scope='sgc_ll_5',
#             is_training=is_training,
#             bn_decay=bn_decay)
#
#         "sum/average all nodes of each graph"
#         net5 = layers.gather(
#             net4,
#             size_x,
#         )
#
#         end_points['lap'] = lap3
#         end_points['res_lap'] = res_lap3
#         end_points['output_seg'] = net4
#         end_points['regs_list'] = [tf.nn.l2_loss(l) for l in [res_lap1, res_lap3, res_lap4]]
#
#         # Symmetric function: max pooling
#         # net5 = layers.max_pool2d(net4, [num_point, 1],
#         #                          padding='SAME', scope='maxpool')
#         end_points['output_softmax'] = net5
#
#     return net5, end_points


def gcn_chebyshev_survival(data,
                           laplacian,
                           size_x,
                           max_node_n,
                           n_classes,
                           is_training,
                           use_attn=True,
                           bn_decay=None):
    """ Classification AGCN + PointNet, input is BxNx3, output Bx40 """
    # batch_size = data.get_shape()[0].value
    # num_point = data.get_shape()[1].value
    end_points = dict()
    if use_attn:
        attn, an_out = attention_net(data, laplacian, size_x, max_node_n, is_training, bn_decay=None)
    else:
        attn = None
    end_points['attn'] = attn
    end_points['an_out'] = an_out


    with tf.variable_scope('backbone_net', reuse=tf.AUTO_REUSE) as chebyshev:

        net1, res_lap1 = layers.GCN_chebyshev(
            data,
            size_x,
            max_node_n,
            laplacian,
            num_output_channels=256,
            scope='cheby_1',
            bn_decay=bn_decay,
        )
        net2, res_lap2 = layers.GCN_chebyshev(
            net1,
            size_x,
            max_node_n,
            laplacian,
            num_output_channels=128,
            scope='cheby_2',
            bn_decay=bn_decay,
        )
        net3, res_lap3 = layers.GCN_chebyshev(
            net2,
            size_x,
            max_node_n,
            laplacian,
            num_output_channels=n_classes,
            scope='cheby_3',
            bn_decay=bn_decay,
        )
        net4 = layers.gather(
            net3,
            size_x,
            attentions=attn,
        )
        # output_loss = tf.sigmoid(net6)
        output = net4
        end_points['output'] = output
        end_points['regs_list'] = [tf.nn.l2_loss(l) for l in [res_lap1, res_lap2, res_lap3]]
        end_points['graph_feature'] = tf.squeeze(tf.reduce_mean(net2, axis=1))

        # # Symmetric function: max pooling
        # net5 = layers.max_pool2d(net4, [num_point, 1],
        #                          padding='SAME', scope='maxpool')
        to_save_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "backbone_net/")
        saver_op = tf.train.Saver({v.op.name: v for v in to_save_vars})
        end_points['saver'] = saver_op
        end_points['saved_vars'] = to_save_vars

    return end_points


def loss_reg(regs):
    regs = tf.reduce_mean(regs)
    return regs


def mse(pred, gt):
    pred = tf.squeeze(pred)
    mse_losses = tf.losses.mean_squared_error(labels=gt, predictions=pred)
    tf.summary.scalar('mse loss', mse_losses)

    return mse_losses


def seg_pred(output_seg):
    output_seg = tf.squeeze(output_seg)
    seg = tf.nn.softmax(output_seg, dim=-1)
    # seg = tf.reduce_max(seg, axis=-1)
    return seg


def agcn_loss(pred, regs, label, reg_weight=0.1):
    """ pred: B*NUM_CLASSES,
        label: B, """
    one_hot_label = tf.one_hot(label, depth=2)
    classify_loss = tf.losses.softmax_cross_entropy(logits=pred, onehot_labels=one_hot_label)
    # classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)

    regs = tf.reduce_mean(regs)
    tf.summary.scalar('Laplacian l2 loss', regs)

    # # Enforce the transformation as orthogonal matrix
    # transform = end_points['transform']  # BxKxK
    # K = transform.get_shape()[1].value
    # mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0, 2, 1]))
    # mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    # mat_diff_loss = tf.nn.l2_loss(mat_diff)
    # tf.summary.scalar('mat loss', mat_diff_loss)

    return classify_loss + reg_weight * regs


def cross_entropy(pred, label):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)

    return classify_loss


if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((8, 500, 128))
        laplacian = tf.ones((8, 500, 500))
        size_x = tf.ones((8,), dtype=tf.int32)
        outputs = gcn_chebyshev_survival(inputs, laplacian, size_x, 500, 1, tf.constant(True), False)
        print(outputs)
