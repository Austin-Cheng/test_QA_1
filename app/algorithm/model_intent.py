# -*- coding: utf-8 -*- 
# @Time : 2022/5/20 13:50 
# @Author : liqianlan
from typing import List

from sentence_transformers import SentenceTransformer, util

from app.constants.config import INTENT_ANNOTATION_PRED_MAP, GENERAL_INTENT_NAME, GENERAL_INTENT_SPACE, \
    INTENT_MODEL_PATH
from app.constants.model_config import INTENT_MODEL_SUPPORTS
from app.utils.intent import Intent
# from resources.intent.sentence_bert.model_use import predict, load_model


class IntentModelManager:
    def __init__(self):
        self.model = None

    def load_model(self, path):
        """
        Load model file.

        Args:
            path: Intent model file path

        Returns:

        """
        self.model = SentenceTransformer(path)
        return self.model

    def predict(self, question):
        """
        意图预测

        Args:
            question: 问句  list or str

        Returns:

        """
        # INTENT_MODEL_SUPPORTS: 小样本支撑集
        samples = INTENT_MODEL_SUPPORTS
        support = [x[0] for x in samples]
        batch_predict = False
        cnt = 1
        if isinstance(question, list):
            batch_predict = True
            cnt = len(question)
            embeddings = self.model.encode(question + support)
        else:
            embeddings = self.model.encode([question] + support)
        cos_sim = util.cos_sim(embeddings, embeddings)

        # Add all pairs to a list with their cosine similarity score
        if not batch_predict:
            all_sentence_combinations = []
            for i in range(cnt, len(cos_sim) - 1):
                all_sentence_combinations.append([cos_sim[i][0], i])
            all_sentence_combinations = sorted(all_sentence_combinations, key=lambda x: x[0], reverse=True)
            most_similar_index = all_sentence_combinations[0][1] - 1
            most_similarity = all_sentence_combinations[0][0]
            pred_intent = samples[most_similar_index][1]
            return pred_intent, float(most_similarity)

    def get_intent(self, query, slots) -> List[Intent]:
        """
        Recognize intent of query.

        >>> irm = IntentModelManager()
        >>> query = 'abcde'
        >>> slots = []
        >>> res = irm.get_intent(query, slots)
        >>> res[0].intent
        其他
        >>> res[0].search_space
        maintenance

        Args:
            query: question text
            slots: entities extracted in query

        Returns: Intent list

        """

        intent, score = self.predict(query)
        # 把标注标签转换为intent标签
        pred_intent = INTENT_ANNOTATION_PRED_MAP[intent]
        intent_name = pred_intent.split('__')[0]
        search_space = pred_intent.split('__')[-1]
        return [Intent(intent_name, search_space, score=score),
                Intent(GENERAL_INTENT_NAME, GENERAL_INTENT_SPACE, score=1.0)]


MODEL_INTENT_RECOGNIZER = IntentModelManager()
MODEL_INTENT_RECOGNIZER.load_model(INTENT_MODEL_PATH)