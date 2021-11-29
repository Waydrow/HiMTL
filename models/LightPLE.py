"""
Author:
    Zuowu Zheng, wozhengzw@gmail.com

"""

import tensorflow as tf

from deepctr.feature_column import build_input_features, input_from_feature_columns
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.layers.utils import combined_dnn_input, reduce_sum


def LightPLE(dnn_feature_columns, num_experts=3, expert_dnn_hidden_units=(256, 128), tower_dnn_hidden_units=(64,),
         gate_dnn_hidden_units=(), l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
         dnn_activation='relu',
         dnn_use_bn=False, task_types=('binary', 'binary'), task_names=('ctr', 'ctcvr')):
    """Instantiates the Multi-gate Mixture-of-Experts multi-task learning architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param num_experts: integer, number of experts.
    :param expert_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of expert DNN.
    :param tower_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of task-specific DNN.
    :param gate_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of gate DNN.
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task_types: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"`` for regression loss. e.g. ['binary', 'regression']
    :param task_names: list of str, indicating the predict target of each tasks

    :return: a Keras model instance
    """
    num_tasks = len(task_names)
    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if num_experts <= 1:
        raise ValueError("num_experts must be greater than 1")

    if len(task_types) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of task_types")

    for task_type in task_types:
        if task_type not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task_type))

    features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)
    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

    # build expert layer
    expert_outs_cate = []
    expert_outs_customer = []
    for i in range(num_experts):
        expert_network_cate = DNN(expert_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed,
                             name='expert_cate_' + str(i))(dnn_input)
        expert_network_customer = DNN(expert_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed,
                             name='expert_customer_' + str(i))(dnn_input)
        expert_outs_cate.append(expert_network_cate)
        expert_outs_customer.append(expert_network_customer)

    expert_concat = []
    expert_concat.append(tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(expert_outs_cate))  # None,num_experts,dim
    expert_concat.append(tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(expert_outs_customer))  # None,num_experts,dim
    expert_concat_ad = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(expert_outs_cate + expert_outs_customer)

    mmoe_outs = []
    # build gate layers for cate, customer
    for i in range(num_tasks - 1):  # one mmoe layer: nums_tasks = num_gates
        # build gate layers
        task_index = i + 1
        gate_input = DNN(gate_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed,
                         name='gate_' + task_names[task_index])(dnn_input)
        gate_out = tf.keras.layers.Dense(num_experts, use_bias=False, activation='softmax',
                                         name='gate_softmax_' + task_names[task_index])(gate_input)
        gate_out = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(gate_out)

        # gate multiply the expert
        gate_mul_expert = tf.keras.layers.Lambda(lambda x: reduce_sum(x[0] * x[1], axis=1, keep_dims=False),
                                                 name='gate_mul_expert_' + task_names[task_index])([expert_concat[i], gate_out])
        mmoe_outs.append(gate_mul_expert)
    # build gate layers for ctr
    gate_input = DNN(gate_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed,
                     name='gate_' + task_names[0])(dnn_input)
    gate_out = tf.keras.layers.Dense(num_experts*2, use_bias=False, activation='softmax',
                                     name='gate_softmax_' + task_names[0])(gate_input)
    gate_out = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(gate_out)

    # gate multiply the expert
    gate_mul_expert = tf.keras.layers.Lambda(lambda x: reduce_sum(x[0] * x[1], axis=1, keep_dims=False),
                                             name='gate_mul_expert_' + task_names[0])([expert_concat_ad, gate_out])
    mmoe_outs.append(gate_mul_expert)

    task_outs = []
    for task_type, task_name, mmoe_out in zip(task_types, task_names, mmoe_outs):
        # build tower layer
        tower_output = DNN(tower_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed,
                           name='tower_' + task_name)(mmoe_out)

        logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(tower_output)
        output = PredictionLayer(task_type, name=task_name)(logit)
        task_outs.append(output)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=task_outs)
    return model
