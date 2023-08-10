# -*- coding: utf-8 -*- 
# @Time : 2022/5/25 10:58 
# @Author : liqianlan
import pytest
from app.algorithm.dict_extractor import DictExtractor
from app.constants.config import ENTITY_DICT_FILE
from app.utils.mention import Mention


class TestDictExtractor:
    extractor = DictExtractor()

    def test_init(self):
        assert self.extractor.actree is not None

    def test_add_dict(self):
        with pytest.raises(FileNotFoundError) as e:
            self.extractor.add_dict('path/not/exist')
        exec_msg = e.value.args[1]
        assert exec_msg == 'No such file or directory'
        self.extractor.add_dict(ENTITY_DICT_FILE)

    def test_extract(self):
        res1 = self.extractor.extract('AS的异常情况')
        assert res1[0].text == '异常情况'
        assert res1[0].type == 'abnormal_type'
        assert res1[0].start == 3
        assert res1[0].end == 7

        res2 = self.extractor.extract('获取资源及其字段的文档的命令')
        assert res2[0].text == '获取资源及其字段的文档'
        assert res2[0].type == 'function'
        assert res2[0].start == 0
        assert res2[0].end == 11
        assert res2[1].text == '命令'
        assert res2[1].type == 'command_type'
        assert res2[1].start == 12
        assert res2[1].end == 14