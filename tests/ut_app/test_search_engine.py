# -*- coding: utf-8 -*- 
# @Time : 2022/5/31 15:15 
# @Author : liqianlan
import re

from app.handlers.graph_search import it_graph_search
from app.handlers.search_engine import ItMaintenanceSearchEngine
from app.utils.mention import Mention


def intent_pipeline(eg, query):
    query = query.lower()
    query = re.sub(' ', '', query)
    entity_mentions = eg.entity_mention_extractor(query)
    entities = eg.entity_link_function(entity_mentions)
    intents, intent_actions = eg.intent_recognizer(query, 'normal_graph', entities)
    return intents, intent_actions


class TestItMaintenanceSearchEngine:
    eg = ItMaintenanceSearchEngine()

    def test_entity_mention_extractor(self):
        text1 = 'as7.0.1.9升级报错'
        res1 = self.eg.entity_mention_extractor(text1)
        assert res1[0].text == 'as'
        assert res1[0].type == 'software'
        assert res1[0].start == 0
        assert res1[0].end == 2
        assert res1[1].text == '7.0.1.9'
        assert res1[1].type == 'software_version'
        assert res1[1].start == 2
        assert res1[1].end == 9
        assert res1[2].text == '升级报错'
        assert res1[2].type == 'unspecific_abnormal'
        assert res1[2].start == 9
        assert res1[2].end == 13

    def test_entity_link_function(self):
        slots = [Mention('anyshare', 'software', 0, 7)]
        slots = self.eg.entity_link_function(slots)
        assert slots[0].text == 'AS'

        slots = [Mention('podpending', 'abnormal', 0, 10)]
        slots = self.eg.dict_normalize(slots)
        assert slots[0].text == 'Pod 停滞在 Pending 状态'

    def test_intent_recognizer(self):
        res, action_res = intent_pipeline(self.eg, 'Pod CrashLoopBackOff')
        assert res[0].intent == '异常信息message-分析类型、原因、处理方法'
        assert res[1].intent == '查文档'
        assert action_res[0] == it_graph_search.exceptionInfo_intent_search_function

        res, action_res = intent_pipeline(self.eg, 'AS7.0.1.9升级到AS7.0.2.0 主模块升级完成后proton-etcd/etcd起不来')
        assert res[0].intent == '软件升级异常分类原因解决方案查询'
        assert res[1].intent == '查文档'
        assert action_res[0] == it_graph_search.exceptionInfo_intent_search_function

        res, action_res = intent_pipeline(self.eg, '升级报错')
        assert res[0].intent == '升级异常-分析类型、原因、处理方法(无软件名称、无异常情况名称）'
        assert action_res[0] == it_graph_search.intent_search_UpgradeException_allType
        assert res[1].intent == '查文档'

        res, action_res = intent_pipeline(self.eg, 'AS升级过程OPA报错pod not ready')
        assert res[0].intent == '升级异常-分析类型、原因、处理方法(只有软件，无软件版本）'
        assert res[1].intent == '查文档'
        assert action_res[0] == it_graph_search.exceptionInfo_intent_search_function

        res, action_res = intent_pipeline(self.eg, 'AS7019主模块升级报错')
        assert res[0].intent == '升级异常-分析类型、原因、处理方法(无异常情况具体信息）'
        assert res[1].intent == '查文档'
        assert action_res[0] == it_graph_search.intent_search_UpgradeException_softwareVersion

        res, action_res = intent_pipeline(self.eg, 'proton-etcd报错的文章有哪些')
        assert res[0].intent == '异常情况-查文档'
        assert res[1].intent == '异常信息message-分析类型、原因、处理方法'

        res, action_res = intent_pipeline(self.eg, 'AS7019升级到AS7020升级指导')
        assert res[0].intent == '软件升级流程查询'
        assert res[1].intent == '查文档'
        assert action_res[0] == it_graph_search.intent_search_UpdateProcedures

        res, action_res = intent_pipeline(self.eg, 'AS常见异常分类有哪些、都有什么异常现象？分别如何解决')
        assert res[0].intent == '异常列表'
        assert res[1].intent == '查文档'
        assert action_res[0] == it_graph_search.intent_search_exceptionList

        res, action_res = intent_pipeline(self.eg, 'AS最新版本是什么')
        assert res[0].intent == '软件版本推理-最新'
        assert res[1].intent == '查文档'
        assert action_res[0] == it_graph_search.intent_search_softwareVersionLatest

        res, action_res = intent_pipeline(self.eg, 'AS有几个版本')
        assert res[0].intent == '软件版本推理-统计'
        assert res[1].intent == '查文档'
        assert action_res[0] == it_graph_search.intent_search_softwareVersionCount

        res, action_res = intent_pipeline(self.eg, 'kubectl命令的选项有哪些')
        assert res[0].intent == '命令描述'
        assert res[1].intent == '查文档'
        res, action_res = intent_pipeline(self.eg, 'kubectl get pod -n中的-n是什么意思')
        assert res[0].intent == '命令参数解释'
        assert res[1].intent == '查文档'
        res, action_res = intent_pipeline(self.eg, '虚拟化技术')
        assert res[0].intent == '名词解释'
        assert res[1].intent == '查文档'

        res, action_res = intent_pipeline(self.eg, '查看AS节点的命令是什么')
        assert res[0].intent == '通过作用查询命令'
        assert res[1].intent == '查文档'

        res, action_res = intent_pipeline(self.eg, '今天的天气怎么样')
        assert res[0].intent == '其他'
        assert res[1].intent == '查文档'


