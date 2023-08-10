# -*- coding: utf-8 -*- 
# @Time : 2022/5/20 13:22 
# @Author : liqianlan
from app.utils.intent import Intent
from .graph_search import it_graph_search

class ActionManager:
    def __init__(self):
        """
        Manage the actions of intent.

        intent_action example:
        {
            'maintenance_graph': {
                '升级流程查询': func1,
                '命令查询': func2,
            },
            'concept_graph': {
                '概念查询': func3
            }
        }
        """
        self.intent_action = {}

    def add_action(self, intent_name, search_space, func):
        """
        把查询逻辑function和意图对应起来，保存为self.intent_action: dict

        Args:
            intent_name: Intent name.
            search_space: Which graph to search.
            func:

        Returns:

        """
        if search_space not in self.intent_action.keys():
            self.intent_action[search_space] = {}
        if intent_name not in self.intent_action[search_space].keys():
            self.intent_action[search_space][intent_name] = {}
        self.intent_action[search_space][intent_name] = func

    def process(self, graph_connector, intents, query, slots):
        """
        根据query识别的多个意图，进行多个意图的查询执行

        Args:
            graph_connector: 图谱连接器
            intents: Intent实例列表
            query: 问句
            slots: 实体识别和消歧结果

        Returns: Answer.

        """
        results = []
        for intent in intents:
            search_space = intent.search_space
            intent_name = intent.intent
            results.append(self.intent_action[search_space][intent_name](graph_connector, search_space,
                                                                         slots))
        pass

    def get_action(self, intent: Intent):
        """
        Get action of an intent.
        Args:
            intent: Intent list, contains intent name and search space.

        Returns: Action function.

        """
        if intent.search_space not in self.intent_action:
            raise ValueError("The intent space has not a related action:{}, intent_name={}".format(
                intent.search_space, intent.intent))
        space = self.intent_action[intent.search_space]
        if intent.intent not in space:
            raise ValueError("The intent has not a related action:{}".format(intent.intent))

        return space[intent.intent]

AM = ActionManager()
# OK pod CrashLoopBackOff
AM.add_action('软件升级异常分类原因解决方案查询', '运维', it_graph_search.UpgradeexceptionInfo_two_software_intent_search)
# OK 升级报错
AM.add_action('升级异常-分析类型、原因、处理方法(无软件名称、无异常情况名称）', '运维', it_graph_search.intent_search_UpgradeException_allType)
# OK AS升级过程策略引擎/OPA报错pod not ready
AM.add_action('升级异常-分析类型、原因、处理方法(只有软件，无软件版本）', '运维', it_graph_search.exceptionInfo_onlySoftware)
# OK AS7.0.1.9主模块升级报错
AM.add_action('升级异常-分析类型、原因、处理方法(无异常情况具体信息）', '运维', it_graph_search.intent_search_UpgradeException_softwareVersion)
# OK AS7.0.1.9主模块升级报错
AM.add_action('软件版本异常-分析类型、原因、处理方法（只有一个软件版本）', '运维', it_graph_search.intent_search_UpgradeException_softwareVersion)
AM.add_action('异常情况-查文档', '文档', it_graph_search.default_intent_search_function)
# OK Pod 处于 Crashing 或别的不健康状态
AM.add_action('异常情况-分析类型、原因、处理方法（无软件）', '运维', it_graph_search.exceptionInfo_intent_search_function)
# OK Pod 处于 Crashing 或别的不健康状态
AM.add_action('异常信息message-分析类型、原因、处理方法', '运维', it_graph_search.exceptionInfo_intent_search_function)
# DELETE AS7019升级的操作步骤是
AM.add_action('软件升级流程查询', '运维', it_graph_search.intent_search_UpdateProcedures)
# OK AS常见异常分类有哪些、都有什么异常现象？分别如何解决
AM.add_action('异常列表', '运维', it_graph_search.intent_search_exceptionList)
# OK AS7.0.1.9的升级服务是
AM.add_action('升级列表', '运维', it_graph_search.intent_search_upgradeList)
# DELETE 查看AS节点的命令是什么
AM.add_action('通过作用查询命令', '运维', it_graph_search.intent_search_commandQuery)
AM.add_action('命令参数解释', '运维', it_graph_search.default_intent_search_function)
# OK AS最新版本是什么
AM.add_action('软件版本推理-最新', '运维', it_graph_search.intent_search_softwareVersionLatest)
# OK AS有几个版本
AM.add_action('软件版本推理-统计', '运维', it_graph_search.intent_search_softwareVersionCount)
AM.add_action('命令描述', '运维', it_graph_search.default_intent_search_function)
AM.add_action('软件版本介绍', '运维', it_graph_search.default_intent_search_function)
AM.add_action('名词解释', '概念', it_graph_search.intent_search_concept)
AM.add_action('查文档', '文档', it_graph_search.default_intent_search_function)
AM.add_action('其他', '无', it_graph_search.default_intent_search_function)
