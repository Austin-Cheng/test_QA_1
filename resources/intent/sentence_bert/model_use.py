# -*- coding: utf-8 -*- 
# @Time : 2022/6/20 9:12 
# @Author : liqianlan
import json
import random

from sentence_transformers import SentenceTransformer, models,\
    InputExample, losses, util
from torch import nn
from torch.utils.data import DataLoader

from app.constants.model_config import INTENT_MODEL_SUPPORTS


def train(pretrain_path, save_path, train_examples):
    """
    Train model.

    Args:
        pretrain_path: 预训练模型路径
        save_path: 训练好的模型保存路径
        train_examples: 训练样本

    Returns:

    """
    word_embedding_model = models.Transformer(pretrain_path, max_seq_length=256)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256,
                               activation_function=nn.Tanh())

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
    # Define your train dataset, the dataloader and the train loss
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)

    # Tune the model
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)

    model.save(save_path)


# predict intent
def load_model(save_path):
    """
    加载模型

    Args:
        save_path: 模型路径

    Returns:

    """
    model = SentenceTransformer(save_path)
    return model


def predict(question, model, samples):
    """
    意图预测

    Args:
        question: 问句  list or str
        model: 模型  class: SentenceTransformer
        samples: 小样本支撑集

    Returns:

    """
    support = [x[0] for x in samples]
    batch_predict = False
    cnt = 1
    if isinstance(question, list):
        batch_predict = True
        cnt = len(question)
        embeddings = model.encode(question+support)
    else:
        embeddings = model.encode([question]+support)
    cos_sim = util.cos_sim(embeddings, embeddings)

    #Add all pairs to a list with their cosine similarity score
    if not batch_predict:
        all_sentence_combinations = []
        for i in range(cnt, len(cos_sim)-1):
            all_sentence_combinations.append([cos_sim[i][0], i])
        all_sentence_combinations = sorted(all_sentence_combinations, key=lambda x: x[0], reverse=True)
        most_similar_index = all_sentence_combinations[0][1]-1
        most_similarity = all_sentence_combinations[0][0]
        pred_intent = samples[most_similar_index][1]
        return pred_intent, float(most_similarity)

    pred_intents = []
    pred_sims = []
    for j in range(cnt):
        all_sentence_combinations = []
        for i in range(cnt, len(cos_sim)-1):
            all_sentence_combinations.append([cos_sim[i][j], i, j])
            all_sentence_combinations = sorted(all_sentence_combinations, key=lambda x: x[0], reverse=True)

            most_similar_index = all_sentence_combinations[0][1] - cnt
            most_similarity = all_sentence_combinations[0][0]
            pred_intent = samples[most_similar_index][1]
            pred_intents.append(pred_intent)
            pred_sims.append(most_similarity)
    return pred_intents, pred_sims


def gather_support(train_file, k):
    """
    选择支撑集

    Args:
        train_file: 标注数据文件
        k: k samples for C categories,每个类别选择k个样本

    Returns:

    """
    with open(train_file, 'r',
              encoding='utf-8') as f:
        lines = f.readlines()

    query_dic = {}
    for line in lines:
        dic = json.loads(line.strip())
        cat = dic['cats'][0]
        text = dic['text']
        if cat not in query_dic:
            query_dic[cat] = []
        query_dic[cat].append(text)

    tmp = [(random.sample(val, k), key)
           for key, val in query_dic.items()]

    samples = [(y, x[1]) for x in tmp for y in x[0]]
    for sample in samples:
        print('("' + sample[0] + '", "' + sample[1] + '"),')
    return samples


if __name__ == '__main__':
    pretrain_path = 'data/chinese_ext_pytorch'
    save_path = 'output/intent_model'

    with open('D:\\PycharmProjects\\IT_maintenance_search\\train.jsonl', 'r',
              encoding='utf-8') as f:
        lines = f.readlines()

    query_dic = {}
    for line in lines:
        dic = json.loads(line.strip())
        cat = dic['cats'][0]
        text = dic['text']
        if cat not in query_dic:
            query_dic[cat] = []
        query_dic[cat].append(text)

    tmp = [(random.sample(val, 3), key)
           for key, val in query_dic.items() if len(val) >= 5]

    samples = [(y, x[1]) for x in tmp for y in x[0]]
    for sample in samples:
        print('("' + sample[0] + '", "' + sample[1] + '"),')
    # 组合训练样本，两两query组成一个训练样本
    train_examples = []
    for i in range(len(samples)):
        for j in range(i, len(samples)):
            train_examples.append(InputExample(
                texts=[samples[i][0], samples[j][0]], label=1.0 if samples[i][1] == samples[j][1] else 0.0))

    # train(pretrain_path, save_path, train_examples)

    model = load_model(save_path)
    res = predict('AS升级后，mongodb的pod异常', model, INTENT_MODEL_SUPPORTS)

    print(res)





