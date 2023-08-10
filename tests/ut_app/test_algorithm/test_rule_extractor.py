# -*- coding: utf-8 -*- 
# @Time : 2022/5/25 14:13 
# @Author : liqianlan
import pytest
from app.algorithm.rule_extractor import RuleExtractor, Rule
from app.constants.config import ENTITY_RULE_FILE
from app.utils.mention import Mention


class TestRule:
    rule = Rule('AS[0-9][.]?[0-9][.]?[0-9][.]?[0-9]\t软件版本')

    def test_init(self):
        assert self.rule.reg == 'AS[0-9][.]?[0-9][.]?[0-9][.]?[0-9]'

        false_rule = 'AS[0-9][.]?[0-9][.]?[0-9][.]?[0-9]\t\t软件版本'
        with pytest.raises(ValueError) as e:
            Rule(false_rule)

        exec_msg = e.value.args[0]
        assert exec_msg == 'The rule is not valid:{}'.format(false_rule)

    def test_fit(self):
        res = self.rule.fit('AS7019的介绍')
        assert res[0].text == 'AS7019'
        assert res[0].type == '软件版本'
        assert res[0].start == 0
        assert res[0].end == 6


class TestRuleExtractor:
    extractor = RuleExtractor()

    def test_init(self):
        assert self.extractor.rule_file is None
        assert not self.extractor.rules

    def test_set_rule_file(self):
        self.extractor.set_rule_file(ENTITY_RULE_FILE)

    def test_extract(self):
        res1 = self.extractor.extract('podnotready怎么办')
        assert res1[0].text == 'podnotready'
        assert res1[0].type == 'abnormal'
        assert res1[0].start == 0
        assert res1[0].end == 11
