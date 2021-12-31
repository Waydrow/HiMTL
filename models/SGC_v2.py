"""
Author:
    Zuowu Zheng, wozhengzw@gmail.com

1. cate, advertiser task 不使用ad features
2. clk = w1*clk + w2*cate + w3*customer
3. embedding TransR

"""

import tensorflow as tf
import numpy as np

from collections import namedtuple, OrderedDict
from deepctr.feature_column import build_input_features, input_from_feature_columns
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.layers.utils import combined_dnn_input, reduce_sum
from deepctr.feature_column import SparseFeat
from deepctr.inputs import create_embedding_matrix, embedding_lookup, get_embedding_vec_list
from tensorflow.python.keras.layers import Layer, Input, Lambda, Add

def get_embedding_vec_list_drop(embedding_dict, input_dict, sparse_feature_columns, return_feat_list=(), drop_feat_list=()):
	embedding_vec_list = []
	for fg in sparse_feature_columns:
		feat_name = fg.name
		if len(return_feat_list) == 0 or feat_name in return_feat_list:
			if len(drop_feat_list) == 0 or feat_name not in drop_feat_list:
				lookup_idx = input_dict[feat_name]
				embedding_vec_list.append(embedding_dict[feat_name](lookup_idx))

	return embedding_vec_list

class TransRLayer(Layer):
	def __init__(self, embedding_dim, embed_reg=1e-4):
		super(TransRLayer, self).__init__()

		self.relation_total = 2

		self.r1 = self.add_weight(name='relation_emb_1', shape=(1,embedding_dim),
									 dtype=tf.float32,
									 initializer=tf.keras.initializers.RandomNormal(),
									 regularizer=tf.keras.regularizers.l2(embed_reg),
									 trainable=True)
		self.r2 = self.add_weight(name='relation_emb_2', shape=(1, embedding_dim),
								  dtype=tf.float32,
								  initializer=tf.keras.initializers.RandomNormal(),
								  regularizer=tf.keras.regularizers.l2(embed_reg),
								  trainable=True)

		rel_matrix = np.zeros([self.relation_total, embedding_dim * embedding_dim], dtype=np.float32)
		for i in range(self.relation_total):
			for j in range(embedding_dim):
				for k in range(embedding_dim):
					if j == k:
						rel_matrix[i][j * embedding_dim + k] = 1.0
		self.rel_matrix = tf.Variable(rel_matrix, name="rel_matrix", dtype=tf.float32)
		self.rel_matrix = tf.identity(self.rel_matrix)

		self.emb_dim = embedding_dim

	def build(self, input_shape):
		super(TransRLayer, self).build(input_shape)

	def call(self, inputs, **kwargs):
		ph, pt1, pt2, nt1, nt2 = inputs
		ph = tf.reshape(ph, [-1, self.emb_dim, 1])
		pt1 = tf.reshape(pt1, [-1, self.emb_dim, 1])
		pt2 = tf.reshape(pt2, [-1, self.emb_dim, 1])
		nt1 = tf.reshape(nt1, [-1, self.emb_dim, 1])
		nt2 = tf.reshape(nt2, [-1, self.emb_dim, 1])
		r1 = tf.tile(self.r1, [tf.shape(ph)[0], 1])
		r2 = tf.tile(self.r2, [tf.shape(ph)[0], 1])

		r1_tmp = tf.zeros([tf.shape(ph)[0],], dtype=tf.int64)
		r2_tmp = tf.ones([tf.shape(ph)[0],], dtype=tf.int64)
		pos_matrix_1 = tf.reshape(tf.nn.embedding_lookup(self.rel_matrix, r1_tmp), [-1, self.emb_dim, self.emb_dim])
		pos_matrix_2 = tf.reshape(tf.nn.embedding_lookup(self.rel_matrix, r2_tmp), [-1, self.emb_dim, self.emb_dim])
		pos_matrix_1 = tf.cast(pos_matrix_1, dtype=tf.float32)
		pos_matrix_2 = tf.cast(pos_matrix_2, dtype=tf.float32)

		ph1 = tf.nn.l2_normalize(tf.reshape(tf.matmul(pos_matrix_1, ph), [-1, self.emb_dim]), 1)
		ph2 = tf.nn.l2_normalize(tf.reshape(tf.matmul(pos_matrix_2, ph), [-1, self.emb_dim]), 1)
		pt1 = tf.nn.l2_normalize(tf.reshape(tf.matmul(pos_matrix_1, pt1), [-1, self.emb_dim]), 1)
		pt2 = tf.nn.l2_normalize(tf.reshape(tf.matmul(pos_matrix_2, pt2), [-1, self.emb_dim]), 1)
		nt1 = tf.nn.l2_normalize(tf.reshape(tf.matmul(pos_matrix_1, nt1), [-1, self.emb_dim]), 1)
		nt2 = tf.nn.l2_normalize(tf.reshape(tf.matmul(pos_matrix_2, nt2), [-1, self.emb_dim]), 1)

		pos1 = tf.reduce_sum((ph1 + r1 - pt1) ** 2, 1, keep_dims = True)
		pos2 = tf.reduce_sum((ph2 + r2 - pt2) ** 2, 1, keep_dims=True)
		neg1 = tf.reduce_sum((ph1 + r1 - nt1) ** 2, 1, keep_dims = True)
		neg2 = tf.reduce_sum((ph2 + r2 - nt2) ** 2, 1, keep_dims=True)

		loss1 = tf.reduce_sum(tf.maximum(pos1 - neg1 + 1, 0))
		loss2 = tf.reduce_sum(tf.maximum(pos2 - neg2 + 1, 0))
		loss = (loss1 + loss2) / 2

		return loss

	def compute_output_shape(self, input_shape):
		return (None, 1)

	def compute_mask(self, inputs, mask=None):
		return None

class FuseOutputLayer(Layer):
	def __init__(self, embed_reg=1e-4):
		super(FuseOutputLayer, self).__init__()

		self.W = self.add_weight(name='w_clk_cate_customer', shape=(1,3),
									 dtype=tf.float32,
									 initializer=tf.keras.initializers.RandomNormal(),
									 regularizer=tf.keras.regularizers.l2(embed_reg),
									 trainable=True)


	def build(self, input_shape):
		super(FuseOutputLayer, self).build(input_shape)

	def call(self, inputs, **kwargs):
		x = tf.concat(inputs, axis=-1) # (None, 3)
		weight = tf.nn.softmax(self.W)
		y = tf.reduce_sum(weight * x, keep_dims=True, axis=1)
		return y

	def compute_output_shape(self, input_shape):
		return (None, 1)

	def compute_mask(self, inputs, mask=None):
		return None

# Structural Gate Control
def SGC_v2(dnn_feature_columns, embedding_dim, shared_expert_num=1, specific_expert_num=1, relation_num = 2, num_levels=2,
		expert_dnn_hidden_units=(256,), tower_dnn_hidden_units=(64,), gate_dnn_hidden_units=(),
		l2_reg_embedding=0.00001,
		l2_reg_dnn=0, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False,
		task_types=('binary', 'binary'), task_names=('ctr', 'ctcvr'), no_ad_features=[], neg_features_tag=[]):
	"""Instantiates the multi level of Customized Gate Control of Progressive Layered Extraction architecture.

	:param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
	:param shared_expert_num: integer, number of task-shared experts.
	:param specific_expert_num: integer, number of task-specific experts.
	:param num_levels: integer, number of CGC levels.
	:param expert_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of expert DNN.
	:param tower_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of task-specific DNN.
	:param gate_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of gate DNN.
	:param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector.
	:param l2_reg_dnn: float. L2 regularizer strength applied to DNN.
	:param seed: integer ,to use as random seed.
	:param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
	:param dnn_activation: Activation function to use in DNN.
	:param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN.
	:param task_types: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"`` for regression loss. e.g. ['binary', 'regression']
	:param task_names: list of str, indicating the predict target of each tasks

	:return: a Keras model instance.
	"""
	num_tasks = len(task_names)
	if num_tasks <= 1:
		raise ValueError("num_tasks must be greater than 1")

	if len(task_types) != num_tasks:
		raise ValueError("num_tasks must be equal to the length of task_types")

	for task_type in task_types:
		if task_type not in ['binary', 'regression']:
			raise ValueError("task must be binary or regression, {} is illegal".format(task_type))

	features = build_input_features(dnn_feature_columns)
	neg_features_input = OrderedDict()
	for fc in neg_features_tag:
		neg_features_input[fc] = Input(shape=(1,), name='neg_'+fc)

	inputs_list = list(features.values()) + list(neg_features_input.values())

	sparse_feature_columns = list(
		filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

	embedding_matrix_dict = create_embedding_matrix(sparse_feature_columns, l2_reg_embedding, seed, prefix='',
													seq_mask_zero=True)

	# sparse_embedding_list = embedding_lookup(embedding_matrix_dict, features, sparse_feature_columns, to_list=True)
	no_trans_embedding_list = get_embedding_vec_list_drop(embedding_matrix_dict, features, sparse_feature_columns,
														  drop_feat_list=['adgroup_id', 'cate_id', 'customer'])

	pos_embedding_list = get_embedding_vec_list(embedding_matrix_dict, features, sparse_feature_columns,
											 return_feat_list=['adgroup_id', 'cate_id', 'customer'])


	no_ad_embedding_list = embedding_lookup(embedding_matrix_dict, features,
											sparse_feature_columns, to_list=True,
											return_feat_list=no_ad_features)

	neg_embedding_list = embedding_lookup(embedding_matrix_dict, neg_features_input,
										  sparse_feature_columns, to_list=True,
										  return_feat_list=neg_features_tag)

	trans_loss = TransRLayer(embedding_dim)(pos_embedding_list + neg_embedding_list)
	sparse_embedding_list = no_trans_embedding_list + pos_embedding_list

	dnn_input = combined_dnn_input(sparse_embedding_list, [])
	no_ad_input = combined_dnn_input(no_ad_embedding_list, [])

	# single Extraction Layer
	def sgc_net(inputs, level_name, is_last=False):
		# inputs: [task1, task2, task3, shared task1 (for task1,2), shared task2 (for task 1,3)]
		specific_expert_outputs = []
		# build task-specific expert layer
		for i in range(num_tasks):
			for j in range(specific_expert_num):
				expert_network = DNN(expert_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,
									 seed=seed,
									 name=level_name + 'task_' + task_names[i] + '_expert_specific_' + str(j))(
					inputs[i])
				specific_expert_outputs.append(expert_network)

		# build task-shared expert layer
		shared_expert_outputs_dic = {}
		for i in range(relation_num):
			shared_expert_outputs_dic[i] = []
			for k in range(shared_expert_num):
				expert_network = DNN(expert_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,
									 seed=seed,
									 name=level_name + 'expert_shared_' + 'relation_' + str(i) + '_' + str(k))(
					inputs[-1*(i+1)])
				shared_expert_outputs_dic[i].append(expert_network)

		# task_specific gate (count = num_tasks)
		cgc_outs = []
		for i in range(num_tasks):
			if i == 0: # main task CTR
				cur_expert_num = specific_expert_num + shared_expert_num * relation_num
				cur_experts = specific_expert_outputs[
							  i * specific_expert_num:(i + 1) * specific_expert_num] + \
							  shared_expert_outputs_dic[i] + shared_expert_outputs_dic[i+1]
			else:
				# concat task-specific expert and task-shared expert
				cur_expert_num = specific_expert_num + shared_expert_num
				# task_specific + task_shared
				cur_experts = specific_expert_outputs[
							  i * specific_expert_num:(i + 1) * specific_expert_num] + shared_expert_outputs_dic[i-1]

			expert_concat = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(cur_experts)

			# build gate layers
			gate_input = DNN(gate_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,
							 seed=seed,
							 name=level_name + 'gate_specific_' + task_names[i])(
				inputs[i])  # gate[i] for task input[i]
			gate_out = tf.keras.layers.Dense(cur_expert_num, use_bias=False, activation='softmax',
											 name=level_name + 'gate_softmax_specific_' + task_names[i])(gate_input)
			gate_out = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(gate_out)

			# gate multiply the expert
			gate_mul_expert = tf.keras.layers.Lambda(lambda x: reduce_sum(x[0] * x[1], axis=1, keep_dims=False),
													 name=level_name + 'gate_mul_expert_specific_' + task_names[i])(
				[expert_concat, gate_out])
			cgc_outs.append(gate_mul_expert)

		# task_shared gate, if the level not in last, add one shared gate
		if not is_last:
			for i in range(relation_num):
				cur_expert_num = num_tasks * specific_expert_num + shared_expert_num
				cur_experts = specific_expert_outputs + shared_expert_outputs_dic[i]
			# cur_experts = specific_expert_outputs + shared_expert_outputs  # all the expert include task-specific expert and task-shared expert

				expert_concat = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(cur_experts)

				# build gate layers
				gate_input = DNN(gate_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,
								 seed=seed,
								 name=level_name + 'gate_shared_' + str(i))(inputs[-1*(i+1)])  # gate for shared task input

				gate_out = tf.keras.layers.Dense(cur_expert_num, use_bias=False, activation='softmax',
												 name=level_name + 'gate_softmax_shared_' + str(i))(gate_input)
				gate_out = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(gate_out)

				# gate multiply the expert
				gate_mul_expert = tf.keras.layers.Lambda(lambda x: reduce_sum(x[0] * x[1], axis=1, keep_dims=False),
														 name=level_name + 'gate_mul_expert_shared_' + str(i))(
					[expert_concat, gate_out])

				cgc_outs.append(gate_mul_expert)
		return cgc_outs

	# build Progressive Layered Extraction
	ple_inputs = [dnn_input, no_ad_input, no_ad_input, dnn_input, dnn_input]  # [task1, task2, ... taskn, shared task1, shared task2...]
	ple_outputs = []
	for i in range(num_levels):
		if i == num_levels - 1:  # the last level
			ple_outputs = sgc_net(inputs=ple_inputs, level_name='level_' + str(i) + '_', is_last=True)
		else:
			ple_outputs = sgc_net(inputs=ple_inputs, level_name='level_' + str(i) + '_', is_last=False)
			ple_inputs = ple_outputs

	logit_outs = []
	task_outs = []
	for task_type, task_name, ple_out in zip(task_types, task_names, ple_outputs):
		# build tower layer
		tower_output = DNN(tower_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed,
						   name='tower_' + task_name)(ple_out)
		logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(tower_output)
		logit_outs.append(logit)
	y_clk = FuseOutputLayer()(logit_outs)
	logit_outs_new = [y_clk, logit_outs[1], logit_outs[2]]
	for task_type, task_name, logit_out in zip(task_types, task_names, logit_outs_new):
		output = PredictionLayer(task_type, name=task_name)(logit_out)
		task_outs.append(output)

	model = tf.keras.models.Model(inputs=inputs_list, outputs=task_outs)
	model.add_loss(1.0 * trans_loss)
	return model
