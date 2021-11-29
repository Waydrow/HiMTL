import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping

from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from deepctr.models import DeepFM, PNN
from util import *


if __name__ == "__main__":

	train, test, train_model_input, test_model_input, dnn_feature_columns = read_data('binary')

	print('train len: %d\ttest_len: %d' % (train.shape[0], test.shape[0]))

	print('Start train PNN')


	# 4.Define Model,train,predict and evaluate
	model = PNN(dnn_feature_columns)
	model.compile("adam", "binary_crossentropy")

	es = EarlyStopping(monitor='val_loss', patience=2)

	history = model.fit(train_model_input, train['clk'].values, batch_size=1024, epochs=20, verbose=2,
						validation_split=0.2, callbacks=[es])
	pred_ans = model.predict(test_model_input, batch_size=1024) # (None, 1)


	print('Test Loss', round(log_loss(test['clk'].values, pred_ans[:, 0]), 4))
	print("Test AUC", round(roc_auc_score(test['clk'].values, pred_ans[:, 0]), 4))
