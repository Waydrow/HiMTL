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


cate_dic_pd = pd.DataFrame(columns=['counter', 'label'])

idx = 0
print('total num: 1550732')
for name, group in data.groupby(['userid', 'cate_id']):
    group = group.sort_values(by='time_stamp')
    sum_tmp = 0
    for index, row in group.iterrows():
        if index == 0:
            label_tmp = 0
        else:
            label_tmp = sum_tmp
        cate_dic_pd = cate_dic_pd.append([{'counter': int(row['counter']), 'label': int(label_tmp)}], ignore_index=True)
        sum_tmp += row['clk']
    if idx % 10000 == 0:
        print('finish ' + str(idx))
    idx += 1

pd.to_pickle(cate_dic_pd, ROOT_DATA+'cate_dic_pd.pkl')