import os
import torch

data_dir = os.getcwd() + '/data/clue/'
train_dir = data_dir + 'train.npz'
test_dir = data_dir + 'test.npz'
files = ['train', 'test']
bert_model = './pretrained_bert_models/bert-base-chinese'
roberta_model = 'pretrained_bert_models/chinese_roberta_wwm_large_ext/'
model_dir = os.getcwd() + '/experiments/clue/'
log_dir = model_dir + 'train.log'
case_dir = os.getcwd() + '/case/bad_case.txt'

# 训练集、验证集划分比例
dev_split_size = 0.1

# 是否加载训练好的NER模型
load_before = False

# 是否对整个BERT进行fine tuning
full_fine_tuning = True

lstm_embedding_size = 768
lstm_dropout_prob = 0.5
hidden_dropout_prob = 0.1
hidden_size = 768

# hyper-parameter
learning_rate = 3e-5
weight_decay = 0.01
clip_grad = 5

batch_size = 32
epoch_num = 5
min_epoch_num = 1
patience = 0.0002
patience_num = 10

gpu = ''

if gpu != '':
    device = torch.device(f"cuda:{gpu}")
else:
    device = torch.device("cpu")

labels = ['异常情况','异常类别','异常原因','软件','软件版本','功能','命令','命令描述','命令参数',
          '命令参数描述','解决方案','软件升级流程','实体类型_命令','实体类型_命令参数','实体类型_命令描述','实体类型_命令参数描述',
          '实体类型_异常原因','实体类型_异常现象','实体类型_类别','实体类型_解决方案','实体类型_软件升级流程','实体类型_功能','实体类型_软件',
          '实体类型_软件版本','实体类型_软件升级服务','非具体异常现象']

label2id = {
    "O": 0,
    "B-异常情况": 1,
    "B-异常类别": 2,
    "B-异常原因": 3,
    "B-软件": 4,
    "B-软件版本": 5,
    "B-功能": 6,
    "B-命令": 7,
    "B-命令描述": 8,
    "B-命令参数": 9,
    "B-命令参数描述": 10,
    "B-解决方案": 11,
    "B-软件升级流程": 12,
    "B-实体类型_命令": 13,
    "B-实体类型_命令参数": 14,
    "B-实体类型_命令描述": 15,
    "B-实体类型_命令参数描述": 16,
    "B-实体类型_异常原因": 17,
    "B-实体类型_异常现象": 18,
    "B-实体类型_类别": 19,
    "B-实体类型_解决方案": 20,
    "B-实体类型_软件升级流程": 21,
    "B-实体类型_功能": 22,
    "B-实体类型_软件": 23,
    "B-实体类型_软件版本": 24,
    "B-实体类型_软件升级服务": 25,
    "B-非具体异常现象": 26,
    "I-异常情况": 27,
    "I-异常类别": 28,
    "I-异常原因": 29,
    "I-软件": 30,
    "I-软件版本": 31,
    "I-功能": 32,
    "I-命令": 33,
    "I-命令描述": 34,
    "I-命令参数": 35,
    "I-命令参数描述": 36,
    "I-解决方案": 37,
    "I-软件升级流程": 38,
    "I-实体类型_命令": 39,
    "I-实体类型_命令参数": 40,
    "I-实体类型_命令描述": 41,
    "I-实体类型_命令参数描述": 42,
    "I-实体类型_异常原因": 43,
    "I-实体类型_异常现象": 44,
    "I-实体类型_类别": 45,
    "I-实体类型_解决方案": 46,
    "I-实体类型_软件升级流程": 47,
    "I-实体类型_功能": 48,
    "I-实体类型_软件": 49,
    "I-实体类型_软件版本": 50,
    "I-实体类型_软件升级服务": 51,
    "I-非具体异常现象": 52,
    "S-异常情况": 53,
    "S-异常类别": 54,
    "S-异常原因": 55,
    "S-软件": 56,
    "S-软件版本": 57,
    "S-功能": 58,
    "S-命令": 59,
    "S-命令描述": 60,
    "S-命令参数": 61,
    "S-命令参数描述": 62,
    "S-解决方案": 63,
    "S-软件升级流程": 64,
    "S-实体类型_命令": 65,
    "S-实体类型_命令参数": 66,
    "S-实体类型_命令描述": 67,
    "S-实体类型_命令参数描述": 68,
    "S-实体类型_异常原因": 69,
    "S-实体类型_异常现象": 70,
    "S-实体类型_类别": 71,
    "S-实体类型_解决方案": 72,
    "S-实体类型_软件升级流程": 73,
    "S-实体类型_功能": 74,
    "S-实体类型_软件": 75,
    "S-实体类型_软件版本": 76,
    "S-实体类型_软件升级服务": 77,
    "S-非具体异常现象": 78
}

id2label = {_id: _label for _label, _id in list(label2id.items())}
