import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, mean_absolute_error
from tensorflow.python.keras.callbacks import EarlyStopping

from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from deepctr.models import SharedBottom, MMOE, PLE
from models import SGC_v2
from util import *


if __name__ == "__main__":

	task_type = sys.argv[1]

	train, test, train_model_input, test_model_input, dnn_feature_columns, \
		no_ad_features, neg_features_tag = read_data_v2(task_type)

	print('train len: %d\ttest_len: %d' % (train.shape[0], test.shape[0]))

	print('Start train SGC_v2')



	# 4.Define Model,train,predict and evaluate
	model = SGC_v2(dnn_feature_columns, embedding_dim=embedding_dim, task_types=['binary', 'binary', 'binary'],
				task_names=['label_clk', 'label_cate', 'label_customer'], shared_expert_num=2, specific_expert_num=2,
				   no_ad_features=no_ad_features, neg_features_tag=neg_features_tag)
	model.compile("adam", loss=["binary_crossentropy", "binary_crossentropy", 'binary_crossentropy'])

	es = EarlyStopping(monitor='val_label_clk_loss', patience=2)

	history = model.fit(train_model_input, [train['clk'].values, train['label_cate'].values, train['label_customer']],
						batch_size=1024, epochs=50, verbose=2, validation_split=0.2, callbacks=[es])
	pred_ans = model.predict(test_model_input, batch_size=1024) # list, each element is (None, 1)

	test_binary(test, pred_ans)
