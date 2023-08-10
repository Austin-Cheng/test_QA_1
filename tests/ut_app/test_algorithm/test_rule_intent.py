# -*- coding: utf-8 -*- 
# @Time : 2022/5/31 10:43 
# @Author : liqianlan
import re

from app.algorithm.rule_intent import IntentRuleManager
from app.constants.config import INTENT_RULE_FILE
from app.utils.mention import Mention


class TestIntentRuleManager:
    irm = IntentRuleManager()

    def test_init(self):
        assert not self.irm.rules

    def test_load_rules(self):
        self.irm.load_rules(INTENT_RULE_FILE)

    def test_get_intent(self):
        query = 'AS的升级流程'
        slots = [Mention('AS', 'software', 0, 2), Mention('升级流程', 'upstep_type', 3, 7)]
        res = self.irm.get_intent(query, slots)
        assert res[0].intent == '软件升级流程查询'
        assert res[0].search_space == '运维'
        assert res[1].intent == '查文档'
        assert res[1].search_space == '文档'


