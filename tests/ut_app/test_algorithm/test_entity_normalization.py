# -*- coding: utf-8 -*- 
# @Time : 2022/5/31 9:33 
# @Author : liqianlan
from app.algorithm.entity_normalization import EntityNormalization
from app.constants.config import ENTITY_SYNONYM_FILE
from app.utils.mention import Mention


class TestEntityNormalization:
    en = EntityNormalization(ENTITY_SYNONYM_FILE)

    def test_init(self):
        assert self.en.synonym_dict['podpending'] == 'Pod 停滞在 Pending 状态'

    def test_dict_normalize(self):
        slots = [Mention('anyshare', 'software', 0, 7)]
        slots = self.en.dict_normalize(slots)
        assert slots[0].text == 'AS'

        slots = [Mention('podpending', 'abnormal', 0, 10)]
        slots = self.en.dict_normalize(slots)
        assert slots[0].text == 'Pod 停滞在 Pending 状态'
