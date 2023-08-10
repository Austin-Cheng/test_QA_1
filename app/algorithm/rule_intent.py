# -*- coding: utf-8 -*- 
# @Time : 2022/5/19 15:48 
# @Author : liqianlan
import re
import json
from typing import List

from app.constants.config import INTENT_RULE_FILE
from app.utils.intent import Intent
from app.utils.mention import Mention


class IntentRuleManager(object):
    def __init__(self):
        self.rules = []

    def load_rules(self, path):
        """
        Load rule file.

        Args:
            path: Intent rule file path

        Returns:

        """
        with open(path, 'rb')as f:
            intent_rule_json = json.load(f)
        self.rules = intent_rule_json
        self._check_rules()

    def _check_rules(self):
        for intent_dic in self.rules:
            if 'intent' not in intent_dic.keys():
                raise Exception('The intent rule is not valid, lack of key "intent": {}'.format(str(intent_dic)))
            if 'space' not in intent_dic.keys():
                raise Exception('The intent rule is not valid, lack of key "space": {}'.format(str(intent_dic)))
            if 'rules' not in intent_dic.keys():
                raise Exception('The intent rule is not valid, lack of key "rules": {}'.format(str(intent_dic)))
            if 'category' not in intent_dic.keys():
                raise Exception('The intent rule is not valid, lack of key "category": {}'.format(str(intent_dic)))

    def get_intent(self, query:str, slots:List[Mention]) -> List[Intent]:
        """
        Recognize intent of query by rules.Support multiple intents.

        >>> irm = IntentRuleManager()
        >>> irm.load_rules(INTENT_RULE_FILE)
        >>> query = 'Pod CrashLoopBackOff'
        >>> slots = [Mention('Pod CrashLoopBackOff','abnormal',1,10)]
        >>> res = irm.get_intent(query, slots)
        >>> res[0].intent
        '升级异常-分析类型、原因、处理方法(无软件名称、无异常情况名称）'
        >>> res[0].search_space
        '运维'

        Args:
            query: Question text.
            slots: Entities extracted from query.

        Returns: Intent list

        """
        intents = []
        categories = []
        text = generate_text_to_match(query, slots)
        for intent_dic in self.rules:
            intent = intent_dic['intent']
            space = intent_dic['space']
            category = intent_dic['category']
            rules = intent_dic['rules']
            for rule in rules:
                if re.search(rule, text) is not None and category not in categories:
                    intents.append(Intent(intent, space, category))
                    categories.append(category)
                    break
        return intents


def generate_text_to_match(query, slots):
    """
    Fill slot type into query.

    Args:
        query: Question text.
        slots: Entity match.

    Returns:

    """
    for slot in reversed(slots):
        query = query[:slot.start]+'{@'+slot.type+'}'+query[slot.end:]

    return query


RULE_INTENT_RECOGNIZER = IntentRuleManager()
RULE_INTENT_RECOGNIZER.load_rules(INTENT_RULE_FILE)
