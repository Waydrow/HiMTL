# HiMTL
Hierarchical Multi-task Learning for CTR Prediction


## Requirements
Please use `pip install -r requirements.txt` to setup the operating environment in `python3.5`.  
Note that we use [DeepCTR](https://github.com/shenweichen/DeepCTR) package.

## Prepare data

1. Download Alimama Data: [Ad Display/Click Data on Taobao.com](https://tianchi.aliyun.com/dataset/dataDetail?dataId=56)
2. Extract the files into the `data/raw_data` directory
3. Follow the code in [DSIN](https://github.com/shenweichen/DSIN) to preprocess data and generate the sampled data
4. Use the code `gen_data/cate_dic_pd.py` and `gen_data/customer_dic_pd.py` to generate the label for cate and customer, respectively.

## Model training
Run the `python train_[model].py`, for example,

```
python train_MMOE.py binary
```

We support three mode `binary`, `regression`, and `regression_norm` to train model in multi-task framwork.

## Supported models

Traditional CTR models:

* AutoInt
* DIFM
* FwFM
* NFM
* PNN
* DeepFM

Multi-task learning models:

* Shared Bottom
* MMOE
* PLE
* SGC
* LightPLE

SGC and LightPLE are our proposed models, which are variants of PLE.