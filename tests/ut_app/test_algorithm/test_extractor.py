# -*- coding: utf-8 -*-
# @Time : 2022/5/19 16:48 
# @Author : liqianlan
import pytest
from app.algorithm.extractor import Extractor
from app.constants.config import ENTITY_RULE_FILE, ENTITY_DICT_FILE


class TestExtractor:
    extractor = Extractor(ENTITY_RULE_FILE, ENTITY_DICT_FILE)

    def test_init(self):
        assert self.extractor.dict_extractor is not None

    def test_merge(self):
        pass

    def test_extract(self):
        res1 = self.extractor.extract('k8s出现podcrashloopbackoff')
        assert res1[0].text == 'k8s'
        assert res1[0].type == 'software'
        assert res1[0].start == 0
        assert res1[0].end == 3
        assert res1[1].text == 'podcrashloopbackoff'
        assert res1[1].type == 'abnormal'
        assert res1[1].start == 5
        assert res1[1].end == 24


