# -*- coding: utf-8 -*- 
# @Time : 2022/6/13 15:56 
# @Author : liqianlan
import json
import os.path
import re
from typing import List

import pandas as pd
import sklearn
from seqeval.metrics import classification_report, f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from app.constants.config import ENT_TYPE_CN_EN_MAP, PROJECT_DIR, INTENT_ANNOTATION_PRED_MAP, ENT_TYPE_EN_CN_MAP, \
    annotated_intents
from app.handlers.search_engine import ItMaintenanceSearchEngine
from app.utils.mention import Mention

def split_data():
    with open(PROJECT_DIR+'/annotation.jsonl', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    dics = [json.loads(line.strip())for line in lines]
    intents = [dic['cats'][0] for dic in dics]
    x_train,x_test, y_train, y_test = train_test_split(lines, intents, test_size=0.2)
    with open(PROJECT_DIR+'/train.jsonl','w',encoding='utf-8')as f:
        f.writelines(x_train)
    with open(PROJECT_DIR+'/test.jsonl','w',encoding='utf-8')as f:
        f.writelines(x_test)

def annotation2label(query, entities):
    # 标注含空格，预测不含空格，需要对齐
    entities = sorted(entities, key=lambda x: x[0])
    blank = [0]*(len(query)+1)
    for i in range(len(query)):
        if query[i] == ' ':
            blank[i+1] = blank[i]+1
        else:
            blank[i+1] = blank[i]
    # print(query)
    for i, entity in enumerate(entities):
        # print(entity)
        entities[i][0] -= blank[entities[i][0]]
        entities[i][1] -= blank[entities[i][1]]
    query = re.sub(' ', '', query)
    mentions = [Mention(query[e[0]: e[1]], ENT_TYPE_CN_EN_MAP[e[2]], e[0], e[1]) for e in entities]
    # 转label
    labels = mention2label(query, mentions)
    return labels


def mention2label(query:str, entities:List[Mention]):
    """
    mention 转 label list

    Args:
        query:
        entities:

    Returns:

    """
    # 转label
    s = 0
    labels = []
    for i, e in enumerate(entities):
        for j in range(s, e.start):
            labels.append('O')

        for j in range(e.start, e.end):
            if e.end - e.start == 1:
                tag = 'S'
            elif j == e.start:
                tag = 'B'
            else:
                tag = 'I'

            if j >= len(query):
                print("out of range:{},{},{}".format(j, len(query), query))
                continue
            labels.append(tag + '-' + e.type)
        s = e.end
    for j in range(s, len(query)):
        labels.append('O')
    return labels


def evaluate_ner(eval_file_path):
    """

    Args:
        eval_file_path: 标注文件

    Returns: 评测结果
    {
        'entity_type1':{
            'precision': 0.8,
            'recall': 0.8,
            'f1': 0.8,
            },
        'entity_type2':{
        },
        'avg / total':{
        }
    }

    """
    with open(eval_file_path, 'r', encoding='utf-8')as f:
        lines = f.readlines()
    # line: {"id":395,"text":"pod状态异常有哪几种","cats":["异常列表（无具体异常信息）"],"entities":[[0,7,"异常类别"]]}
    y_true = []
    y_pred = []
    tokens = []
    eg = ItMaintenanceSearchEngine()
    for line in lines:
        dic = json.loads(line.strip())
        query = dic['text']
        query_id = dic['id']
        entities = dic['entities']
        labels = annotation2label(query, entities)
        y_true.append(labels)

        query_no_blank = re.sub(' ', '', query)
        query = query.lower()
        query = re.sub(' ', '', query)
        extracted_mentions = eg.entity_mention_extractor(query)
        predictions = mention2label(query, extracted_mentions)
        y_pred.append(predictions)

        tokens += list(query_no_blank)

    df = pd.DataFrame.from_dict({'text': tokens, 'true': sum(y_true, []), 'pred': sum(y_pred, [])})
    fn = os.path.split(eval_file_path)[-1]
    df.to_csv(PROJECT_DIR+'/'+fn[:-6]+'_ner_pred.csv', encoding='gbk')
    report = classification_report(y_true, y_pred, digits=4)
    print(report)
    lines = report.split('\n')
    eval_dict = {}
    for i in range(2, len(lines)):
        line = lines[i]
        vals = list(filter(None, line.split('    ')))
        if not vals:
            continue
        if vals[0].strip() not in ENT_TYPE_EN_CN_MAP:
            name = vals[0].strip()
        else:
            name = ENT_TYPE_EN_CN_MAP[vals[0].strip()]
        eval_dict[name] = {
            'precision': float(vals[1]),
            'recall': float(vals[2]),
            'f1': float(vals[3])
        }

    return eval_dict


def evaluate_intent(eval_file_path):
    """

    Args:
        eval_file_path: 标注文件

    Returns: 评测结果

    Examples:
    {
        'intent_type1':{
            'precision': 0.8,
            'recall': 0.8,
            'f1': 0.8,
            },
        'intent_type2':{
        },
        'weighted avg':{
        }

    }

    """
    with open(eval_file_path, 'r', encoding='utf-8')as f:
        lines = f.readlines()
    # line: {"id":395,"text":"pod状态异常有哪几种","cats":["异常列表（无具体异常信息）"],"entities":[[0,7,"异常类别"]]}
    y_true = []
    y_pred = []
    queries = []
    idxes = []
    eg = ItMaintenanceSearchEngine()
    for line in lines:
        dic = json.loads(line.strip())
        query = dic['text']
        query_id = dic['id']
        label = dic['cats']
        # 此处只筛选标注的5个意图进行评估  标注文件里有个别其他意图query
        if label[0] not in annotated_intents:
            continue
        queries.append(query)
        idxes.append(query_id)
        y_true.append(INTENT_ANNOTATION_PRED_MAP[label[0]])

        query = query.lower()
        query = re.sub(' ', '', query)
        entity_mentions = eg.entity_mention_extractor(query)
        entities = eg.entity_link_function(entity_mentions)
        intents, _ = eg.intent_recognizer(query, 'normal_graph', entities)
        y_pred.append(intents[0].intent+'__'+intents[0].search_space if intents else '')

    df = pd.DataFrame.from_dict({'text': queries, 'true': y_true, 'pred': y_pred})
    fn = os.path.split(eval_file_path)[-1]
    df.to_csv(PROJECT_DIR + '/'+fn[:-6]+'_intent_pred.csv', encoding='gbk')
    report = sklearn.metrics.classification_report(y_true, y_pred)
    print(report)
    lines = report.split('\n')
    eval_dict = {}
    for i in range(2, len(lines)):
        line = lines[i]
        vals = list(filter(None, line.split('    ')))
        if not vals:
            continue
        if vals[0].strip() == 'accuracy' or vals[0].strip() == 'macro avg':
            continue
        eval_dict[vals[0].strip()] = {
            'precision': float(vals[1]),
            'recall': float(vals[2]),
            'f1': float(vals[3])
        }
    return eval_dict
