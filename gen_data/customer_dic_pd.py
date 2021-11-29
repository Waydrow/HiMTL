import os
import numpy as np
import pandas as pd

FRAC = 0.25

ROOT_DATA = '/data/alimama/sampled_data/'

data = pd.read_pickle(ROOT_DATA + 'data_0.25.pkl')
print(data.shape)
label_cate = np.zeros([data.shape[0], 1], dtype=np.int)
label_customer = np.zeros([data.shape[0], 1], dtype=np.int)
data['label_cate']=label_cate
data['label_customer']=label_customer
data['counter'] = range(len(data))

customer_dic_pd = pd.DataFrame(columns=['counter', 'label'])

idx = 0
print('total num: 4495427')
for name, group in data.groupby(['userid', 'customer']):
    group = group.sort_values(by='time_stamp')
    sum_tmp = 0
    for index, row in group.iterrows():
        if index == 0:
            label_tmp = 0
        else:
            label_tmp = sum_tmp
        customer_dic_pd = customer_dic_pd.append([{'counter': int(row['counter']), 'label': int(label_tmp)}], ignore_index=True)
        sum_tmp += row['clk']
    if idx % 100000 == 0:
        print('finish ' + str(idx))
    idx += 1

pd.to_pickle(customer_dic_pd, ROOT_DATA+'customer_dic_pd.pkl')