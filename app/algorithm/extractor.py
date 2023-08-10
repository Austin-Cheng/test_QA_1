# -*- coding: utf-8 -*-
from typing import Dict, Iterable, Union, Any, List, Optional

from app.algorithm.dict_extractor import DictExtractor
from app.algorithm.model_extractor import ModelExtractor
from app.algorithm.rule_extractor import RuleExtractor
from app.constants.config import ENTITY_DICT_FILE, ENTITY_RULE_FILE, NER_MODEL_THR, ENTITY_MODEL_FILE
from app.utils.mention import Mention
from app.utils.util import validate


class Extractor:
    def __init__(self, rule_file=None, dict_file=None, model_file=None):
        """
        实体抽取初始化，导入各个配置文件

        Args:
            rule_file: 规则文件路径
            dict_file: 词典文件路径
            model_file: 模型文件路径
        """
        self.rule_extractor = RuleExtractor()
        self.rule_extractor.set_rule_file(rule_file)
        self.dict_extractor = DictExtractor()
        self.dict_extractor.add_dict(dict_file)
        self.model_extractor = ModelExtractor()
        self.model_extractor.load_model(model_file)
        self.mentions_model = []
        self.mentions_rule = []
        self.mentions_dict = []

    def merge(self):
        """
        融合抽取结果

        Returns:

        """
        for i in range(len(self.mentions_dict)-1, -1, -1):
            if not validate(self.mentions_rule, self.mentions_dict[i]):
                self.mentions_dict.pop(i)
        for i in range(len(self.mentions_model) - 1, -1, -1):
            if self.mentions_model[i].confidence < NER_MODEL_THR \
                    or not validate(self.mentions_rule, self.mentions_model[i])\
                    or not validate(self.mentions_dict, self.mentions_model[i]):
                self.mentions_model.pop(i)

    def extract(self, query: str, **kwargs) -> List[Mention]:
        """
        综合规则词典模型的实体抽取

        >>> query = 'anyshare的升级流程'
        >>> extractor = Extractor(ENTITY_RULE_FILE, ENTITY_DICT_FILE, ENTITY_MODEL_FILE)
        >>> res = extractor.extract(query)
        >>> res
        >>> res[0].text
        'anyshare'
        >>> res[1].text
        '升级流程'

        Args:
            query: 问句

        Returns: 实体列表

        """

        self.mentions_rule = self.rule_extractor.extract(query)
        self.mentions_dict = self.dict_extractor.extract(query)
        self.mentions_model = self.model_extractor.extract(query)
        self.merge()
        mentions = sorted(self.mentions_rule+self.mentions_dict+self.mentions_model,
                          key=lambda x: x.start)
        return mentions


ENTITY_EXTRACTOR = Extractor(ENTITY_RULE_FILE, ENTITY_DICT_FILE, ENTITY_MODEL_FILE)
if __name__ == '__main__':

    res = ENTITY_EXTRACTOR.extract('as升级时，mongodb启动失败的原因是什么')
    for ent in res:
        print(ent)