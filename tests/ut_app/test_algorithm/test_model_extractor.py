# -*- coding: utf-8 -*- 
# @Time : 2022/5/31 19:39 
# @Author : liqianlan
from app.algorithm.model_extractor import ModelExtractor


class TestModelExtractor:
    extractor = ModelExtractor()

    def test_init(self):
        pass

    def test_load_model(self):
        model_file = ''
        self.extractor.load_model(model_file)

    def test_extract(self):
        res1 = self.extractor.extract('podnotready怎么办')
        assert res1[0].text == 'podnotready'
        assert res1[0].type == 'abnormal'
        assert res1[0].start == 0
        assert res1[0].end == 11
