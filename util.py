import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, mean_absolute_error

from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names

embedding_dim = 16
DATA_PATH = '/root/autodl-tmp/data/'

# DATA_PATH = '/Users/wozhengzw/Documents/data/HiMTL/sampled_data/'

def read_data(task_type):
	data = pd.read_pickle(DATA_PATH + 'data_mtl_v5_0.25.pkl')

	sparse_features = ['userid', 'adgroup_id', 'pid', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level',
					   'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level', 'campaign_id',
					   'customer', 'cate_id', 'brand'] # 15 sparse features

	if task_type == 'binary':
		data['label_cate'] = data['label_cate'].apply(lambda x: 1 if x > 0 else 0)
		data['label_customer'] = data['label_customer'].apply(lambda x: 1 if x > 0 else 0)
	elif task_type == 'regression_norm':
		dense_label = ['label_cate', 'label_customer']
		mms = MinMaxScaler(feature_range=(0, 1))
		data[dense_label] = mms.fit_transform(data[dense_label])



	fixlen_feature_columns = [SparseFeat(feat, data[feat].max() + 1, embedding_dim=16) for feat in sparse_features]

	dnn_feature_columns = fixlen_feature_columns

	feature_names = get_feature_names(dnn_feature_columns)

	# 3.generate input data for model

	train = data.loc[data['time_stamp'] < 1494633600]
	test = data.loc[data['time_stamp'] >= 1494633600]

	train_model_input = {name: train[name] for name in feature_names}
	test_model_input = {name: test[name] for name in feature_names}

	return train, test, train_model_input, test_model_input, dnn_feature_columns

# 辅助任务不使用ad features
def read_data_v1(task_type):
	data = pd.read_pickle(DATA_PATH + 'data_mtl_v5_0.25.pkl')

	sparse_features = ['userid', 'adgroup_id', 'pid', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level',
					   'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level', 'campaign_id',
					   'customer', 'cate_id', 'brand'] # 15 sparse features

	no_ad_features = ['userid', 'pid', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level',
					   'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level']

	if task_type == 'binary':
		data['label_cate'] = data['label_cate'].apply(lambda x: 1 if x > 0 else 0)
		data['label_customer'] = data['label_customer'].apply(lambda x: 1 if x > 0 else 0)
	elif task_type == 'regression_norm':
		dense_label = ['label_cate', 'label_customer']
		mms = MinMaxScaler(feature_range=(0, 1))
		data[dense_label] = mms.fit_transform(data[dense_label])



	fixlen_feature_columns = [SparseFeat(feat, data[feat].max() + 1, embedding_dim=16) for feat in sparse_features]

	dnn_feature_columns = fixlen_feature_columns

	# no_ad_feature_columns = [SparseFeat(feat, data[feat].max() + 1, embedding_dim=16) for feat in no_ad_features]

	feature_names = get_feature_names(dnn_feature_columns)

	# 3.generate input data for model

	train = data.loc[data['time_stamp'] < 1494633600]
	test = data.loc[data['time_stamp'] >= 1494633600]

	train_model_input = {name: train[name] for name in feature_names}
	test_model_input = {name: test[name] for name in feature_names}

	return train, test, train_model_input, test_model_input, dnn_feature_columns, no_ad_features

# for SGC_v2
def read_data_v2(task_type):
	data = pd.read_pickle(DATA_PATH + 'data_mtl_v5_0.25.pkl')

	sparse_features = ['userid', 'adgroup_id', 'pid', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level',
					   'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level', 'campaign_id',
					   'customer', 'cate_id', 'brand'] # 15 sparse features

	no_ad_features = ['userid', 'pid', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level',
					   'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level']

	neg_features = ['neg_cate_id', 'neg_customer']
	neg_features_tag = ['cate_id', 'customer']

	if task_type == 'binary':
		data['label_cate'] = data['label_cate'].apply(lambda x: 1 if x > 0 else 0)
		data['label_customer'] = data['label_customer'].apply(lambda x: 1 if x > 0 else 0)
	elif task_type == 'regression_norm':
		dense_label = ['label_cate', 'label_customer']
		mms = MinMaxScaler(feature_range=(0, 1))
		data[dense_label] = mms.fit_transform(data[dense_label])



	dnn_feature_columns = [SparseFeat(feat, data[feat].max() + 1, embedding_dim=16) for feat in sparse_features]

	# neg_feature_columns = [SparseFeat(feat, data[feat].max() + 1, embedding_dim=16) for feat in neg_features]

	feature_names = sparse_features + neg_features

	# 3.generate input data for model

	train = data.loc[data['time_stamp'] < 1494633600]
	test = data.loc[data['time_stamp'] >= 1494633600]

	train_model_input = {name: train[name] for name in feature_names}
	test_model_input = {name: test[name] for name in feature_names}

	return train, test, train_model_input, test_model_input, dnn_feature_columns, no_ad_features, neg_features_tag

def test_binary(test, pred_ans):
	# pred_ans = pred_ans.astype(np.float64)
	print('Test clk Loss', round(log_loss(test['clk'].values, pred_ans[0][:, 0], eps=1e-7), 4))
	print("Test clk AUC", round(roc_auc_score(test['clk'].values, pred_ans[0][:, 0]), 4))

	print('Test cate Loss', round(log_loss(test['label_cate'].values, pred_ans[1][:, 0], eps=1e-7), 4))
	print("Test cate AUC", round(roc_auc_score(test['label_cate'].values, pred_ans[1][:, 0]), 4))

	print('Test customer Loss', round(log_loss(test['label_customer'].values, pred_ans[2][:, 0], eps=1e-7), 4))
	print("Test customer AUC", round(roc_auc_score(test['label_customer'].values, pred_ans[2][:, 0]), 4))


def test_regression(test, pred_ans):
	print('Test clk Loss', round(log_loss(test['clk'].values, pred_ans[0][:, 0]), 4))
	print("Test clk AUC", round(roc_auc_score(test['clk'].values, pred_ans[0][:, 0]), 4))

	print('Test cate MSE', round(mean_squared_error(test['label_cate'].values, pred_ans[1][:, 0]), 4))
	print('Test cate MAE', round(mean_absolute_error(test['label_cate'].values, pred_ans[1][:, 0]), 4))

	print('Test customer MSE', round(mean_squared_error(test['label_customer'].values, pred_ans[2][:, 0]), 4))
	print('Test customer MAE', round(mean_absolute_error(test['label_customer'].values, pred_ans[2][:, 0]), 4))