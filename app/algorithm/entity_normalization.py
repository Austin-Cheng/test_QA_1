# -*- coding: utf-8 -*-
import os
import pickle
import time

import numpy as np
import torch

from transformers import BertTokenizer, BertModel, logging
logging.set_verbosity_error()
from typing import List
# from bert_serving.client import BertClient
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.constants.config import ENTITY_SYNONYM_FILE, BERT_SERVER_IP, ENTITY_LINK_DIR, LINK_SIM_THR, ENTITY_VECTOR_DIR, \
    INTENT_MODEL_PATH, LINK_SIM_THR_LCS, ENTITY_VECTOR_DIR_LOCAL, ENTITY_LINK_MODEL_DIR
from app.utils.mention import Mention
from app.utils.util import get_lcs_len, load_kernel_bias, transform_and_normalize


def encode_by_local_model(texts, model, tokenizer):

    res = []
    for text in texts:
        input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
        # input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(list(mentions[1].text))).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_ids)
        # outputs = model(input_ids)
        last_hidden_state = outputs.last_hidden_state.detach().numpy()
        vectors = np.mean(last_hidden_state, axis=1)
        # hidden_states = outputs[2]
        # print(hidden_states.shape)
        # print(hidden_states)
        res.append(vectors)
    matrix = np.concatenate(res, axis=0)
    return matrix


class EntityNormalization:
    """
    实体消歧
    """
    def __init__(self, synonym_path=None, vector_path=None):
        """
        导入近义词文件，和词向量模型路径

        Args:
            synonym_path: 近义词文件
            vector_path: 词向量模型路径
        """
        self.vector_path = vector_path
        if self.vector_path is not None and os.path.exists(vector_path):
            self.tokenizer = BertTokenizer.from_pretrained(ENTITY_LINK_MODEL_DIR)
            self.model = BertModel.from_pretrained(ENTITY_LINK_MODEL_DIR)
            self.model.eval()
        if synonym_path is None:
            return
        with open(synonym_path, 'r', encoding='utf-8')as f:
            lines = f.readlines()
        word_pairs = [line.strip().split('\t') for line in lines]
        self.synonym_dict = {}
        for i, word_pair in enumerate(word_pairs):
            if len(word_pair) != 2:
                raise ValueError('The synonym pair in line {} is not valid:{}'.format(i+1, word_pair))
            self.synonym_dict[word_pair[0]] = word_pair[1]

    def dict_normalize(self, mentions) -> List[Mention]:
        """
        词典消歧

        >>> slots = [Mention('podpending', 'abnormal', 0, 10)]
        >>> en = EntityNormalization(ENTITY_SYNONYM_FILE)
        >>> res = en.dict_normalize(slots)
        >>> res[0].text
        'Pod 停滞在 Pending 状态'
        >>> res[0].type
        'abnormal'

        Args:
            mentions: 实体抽取结果 List[Mention]

        Returns:

        """
        for mention in mentions:
            if mention.text in self.synonym_dict.keys():
                new_text = self.synonym_dict[mention.text]
                mention.text = new_text
                mention.normalized = True
        return mentions

    def similar_normalize(self, mentions) -> List[Mention]:
        """
        相似度消歧

        >>> slots = [Mention('Runcontainereror', 'abnormal', 0, 10)]
        >>> en = EntityNormalization()
        >>> res = en.similar_normalize(slots)
        >>> res[0].text
        'pod RunContainerError状态'
        >>> res[0].type
        'abnormal'
        >>> res[0].normalized
        True

        Args:
            mentions: 实体抽取结果 List[Mention]

        Returns:

        """
        # kernel, bias = load_kernel_bias(self.vector_path)
        # 本地模型load
        mention_pairs = [(mention, i) for i, mention in enumerate(mentions) if mention.normalized is False]
        mention_texts = [pair[0].text for pair in mention_pairs]
        if not mention_texts:
            return mentions
        vectors = encode_by_local_model(mention_texts, self.model, self.tokenizer)

        # 远程bert server
        # mention_pairs = [(mention, i) for i, mention in enumerate(mentions) if mention.normalized is False]
        # with BertClient(ip=BERT_SERVER_IP) as bc:
        #     # vectors = bc.encode([list(pair[0].text) for pair in mention_pairs],
        #     #                     is_tokenized=True)
        #     vectors = bc.encode([pair[0].text for pair in mention_pairs])

        for vec, pair in zip(vectors, mention_pairs):
            # 和同类型实体计算相似度
            file_path = os.path.join(ENTITY_VECTOR_DIR_LOCAL, pair[0].type+'.pkl')
            if not os.path.exists(file_path):
                continue
            with open(file_path, 'rb')as f:
                entity_vecs = pickle.load(f)
            vec = vec.reshape(1, -1)
            # vec = transform_and_normalize(vec, kernel, bias)
            sim_matrix = cosine_similarity(vec, entity_vecs['vector'])
            sims = list(sim_matrix[0])
            # print(sorted([(x,y) for x, y in zip(sims, entity_vecs['entity'])], reverse=True))
            max_sim_idx = sims.index(max(sims))
            if sims[max_sim_idx] < LINK_SIM_THR:
                continue
            mentions[pair[1]].text = entity_vecs['entity'][max_sim_idx]
            mentions[pair[1]].normalized = True

        return mentions

    def similar_normalize_by_lcs(self, mentions) -> List[Mention]:
        """
        相似度消歧,针对报错信息漏字符等问题
        采用LCS

        >>> slots = [Mention('podpeding', 'abnormal', 0, 10)]
        >>> en = EntityNormalization()
        >>> res = en.similar_normalize_by_lcs(slots)
        >>> res[0].text
        'Pod 停滞在 Pending 状态'
        >>> res[0].type
        'abnormal'
        >>> res[0].normalized
        True

        Args:
            mentions: 实体抽取结果 List[Mention]

        Returns:

        """
        for i, mention in enumerate(mentions):
            if mention.normalized is True:
                continue
            file_path = os.path.join(ENTITY_VECTOR_DIR_LOCAL, mention.type+'.pkl')
            if not os.path.exists(file_path):
                continue
            with open(file_path, 'rb')as f:
                entity_vecs = pickle.load(f)
            entities = entity_vecs['entity']
            max_sim = 0.0
            max_sim_ent = None
            for ent in entities:
                sim = get_lcs_len(mention.text, ent.lower())/len(ent)
                if sim > max_sim:
                    max_sim = sim
                    max_sim_ent = ent
            # print(max_sim, max_sim_ent)
            if max_sim < LINK_SIM_THR_LCS:
                continue

            mentions[i].text = max_sim_ent
            mentions[i].normalized = True
        return mentions

    def normalize(self, mentions) -> List[Mention]:
        """
        综合dict_normalize,similar_normalize函数

        Args:
            mentions: 实体抽取结果 List[Mention]

        Returns: 实体消歧结果 List[Mention]

        """
        mentions = self.dict_normalize(mentions)
        mentions = self.similar_normalize_by_lcs(mentions)
        mentions = self.similar_normalize(mentions)
        return mentions


ENTITY_LINKER = EntityNormalization(ENTITY_SYNONYM_FILE, ENTITY_LINK_MODEL_DIR)
