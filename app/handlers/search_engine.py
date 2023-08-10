# -*- coding: utf-8 -*-
import re
from typing import Iterable

from fastapi.exceptions import FastAPIError

from app.algorithm.entity_normalization import ENTITY_LINKER
from app.algorithm.extractor import ENTITY_EXTRACTOR
from app.algorithm.model_intent import MODEL_INTENT_RECOGNIZER
from app.algorithm.rule_intent import RULE_INTENT_RECOGNIZER
from app.constants import config
from app.handlers.intent_related_action import AM
from app.handlers.model import SearchResultHelper
from app.utils.intent import Intent
from cognition.GraphSearchEngine import GraphSearchEngine


# from app.constants.config import GENERAL_INTENT_NAME, GENERAL_INTENT_SPACE, UNRECOGNIZED_INTENT_SPACE, \
#     UNRECOGNIZED_INTENT_NAME, SearchGraphSpace, INTENT_MODEL_THR


class ItMaintenanceSearchEngine(GraphSearchEngine):
    def __init__(self, graph_connector=None):
        super(ItMaintenanceSearchEngine, self).__init__(graph_connector)

    def entity_mention_extractor(self, query) -> Iterable:
        """
        Extract entity mentions in query.

        Args:
            query: Question text.

        Returns: Mention list.

        """
        # 实体识别方法，获取query中实体文本集合。
        mentions = ENTITY_EXTRACTOR.extract(query)
        return mentions

    def intent_recognizer(self, query, graph_type, entities):
        """
        Recognize the intent of query,containing intent name and search graph.
        Args:
            query: Input query.
            graph_type:
            entities: Entities extarcted from query.

        Returns:

        """
        # TODO: 修改基类的输入
        # 意图识别方法，搜索意图识别
        intents = RULE_INTENT_RECOGNIZER.get_intent(query, entities)
        model_intents = MODEL_INTENT_RECOGNIZER.get_intent(query, entities)
        intent_names = [intent.intent + '__' + intent.search_space for intent in intents]
        for intent in model_intents:
            if intent.score > config.INTENT_MODEL_THR and intent.intent + '__' + intent.search_space not in intent_names:
                intents.append(intent)

        # “其他”意图
        if len(intents) == 1 and intents[0].intent == config.GENERAL_INTENT_NAME and intents[
            0].search_space == config.GENERAL_INTENT_SPACE:
            intents.insert(0, Intent(config.UNRECOGNIZED_INTENT_NAME, config.UNRECOGNIZED_INTENT_SPACE))
        intent_actions = [AM.get_action(intent) for intent in intents]
        return intents, intent_actions

    def entity_link_function(self, entity_metions) -> Iterable:
        """
        Link the mention to real graph entity.

        Args:
            entity_metions: Entities extracted from query.

        Returns:

        """
        # 实体链接方法，将文本中提取到的实体文本集合链接到图中的实体,返回实体信息集合。将实体mention连接到对应的图谱的实体上
        entities = ENTITY_LINKER.normalize(entity_metions)
        return entities

    async def search(self, query: str, **kwargs):
        query = query.lower()
        query = re.sub(' ', '', query)
        entity_mentions = self.entity_mention_extractor(query)
        entities = self.entity_link_function(entity_mentions)
        kwargs.update({'input_entities': entities})
        intents, intent_actions = self.intent_recognizer(query, 'normal_graph', entities)

        search_results = {}
        for intent, intent_action, in zip(intents, intent_actions):
            res = {}
            try:
                res = await intent_action(self.graph_connector, config.SearchGraphSpace(intent.search_space).name,
                                          **kwargs)
            except BaseException:
                raise FastAPIError("query error")
            appendLen = SearchResultHelper.parseAndAppend(search_results, res)
        SearchResultHelper.addQuery(search_results, query)
        return SearchResultHelper.formatSearchResult(search_results, kwargs['offset'], kwargs['limit'], kwargs['graph'],
                                                     kwargs['dt'])

    async def concept(self, query: str, **kwargs):
        """
        划词搜索，概念图谱查询接口
        """
        search_results = SearchResultHelper.genConcept()
        SearchResultHelper.addQuery(search_results, query)
        return SearchResultHelper.formatSearchResult(search_results, 0, 1, kwargs['graph'], 'txt')
