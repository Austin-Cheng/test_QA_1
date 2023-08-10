# -*- coding: utf-8 -*-
import json
import nebula2
from nebula2.gclient.net import ConnectionPool
from nebula2.Config import Config
import pandas as pd
from cognition.GraphSearch.NebulaSearch import NebulaSearch
from typing import Iterable, List, Union, Any

from cognition.GraphSearch.GraphSearch import (
    GraphSearch,
    TagType, Statement, OperateEnum, Path, Vertex, Edge, ALL_VERTEX_TYPE,
    DIRECTION,
    EmptySearchException, InvalidTypeException
)

class ItMaintenanceGraphSearch(NebulaSearch):
    """Other project customize search functions."""

    async def default_intent_search_function(self, graph_connector, space_name, **kwargs):
        return 'just a try'

    async def UpgradeexceptionInfo_two_software_intent_search(self, graph_connector, space_name, **kwargs):
        """
        "软件升级异常分类原因解决方案查询（两个软件版本）"这一意图对应的图谱查询逻辑。
        :param graph_connector: graph db connector
        :param space_name: graph db space name
        :param **kwargs：entity mention extracted in qurey text , e.g. [Mention(text,type,s,e),Mention(text2,type2,s2,e2)]
        :return: json result, 给出引起这个中间状态的异常的原因有哪几种及解决方案
        Examples:
        >>> ItMaintenanceGraphSearch.UpgradeexceptionInfo_two_softwareVision_intent_search(client, 'nebula0526', Mention('Pod 处于 ImagePullBackOff 状态', 'abnormal', 1, 27))
        {'errors': [{'code': 0}], 'results': [{'spaceName': 'nebula0526', 'data': [{'meta': [None, None], 'row': ['从私有镜像仓库拉取镜像', '配置从私有仓库拉取镜像']}, {'meta': [None, None], 'row': ['镜像名错误', '修复镜像名']}, {'meta': [None, None], 'row': ['镜像无效、不存在', '修复tag']}], 'columns': ['abnormal_causes', 'method_disposal'], 'errors': {'code': 0}, 'latencyInUs': 13384}]}
        """
        softwareVersion = []
        abnormal_cate_mention = []
        abnormal_mention = []
        res = {}
        for mention in kwargs.get('input_entities'):
            if mention.type == 'cate':
                abnormal_cate_mention.append(mention.text)
            if mention.type == 'abnormal':
                abnormal_mention.append(mention.text)
            if mention.type == 'software_version':
                softwareVersion.append(mention.text)
        if len(softwareVersion) != 2:
            return {}
        abnormal_cate_ngql = '''
            USE {};
           (MATCH(v0: software_upgrate)-[e0: software_upgrate_2_software_version_f]-(v1:software_version)
            where v1.name == '{}'
            RETURN id(v0) as vid
            INTERSECT
            MATCH(v0: software_upgrate)-[e0: software_upgrate_2_software_version]-(v1:software_version)
            where v1.name == '{}'
            RETURN id(v0) as vid)
            | 
            GO FROM $-.vid OVER software_upgrate_2_abnormal YIELD dst(edge) as abnormal_id, properties($$).name as abnormal
            |
            GO FROM $-.abnormal_id OVER exception_cate_2_abnormal REVERSELY
            WHERE properties($$).name in {}
            YIELD dst(edge) as abnormal_id, $-.abnormal as abnormal, properties($$).name as exception_cate
            | 
            GO FROM $-.abnormal_id OVER abnormal_2_Abnormal_cause YIELD DISTINCT dst(edge) as abnormal_cause_id ,properties($$).name as abnormal_cause, $-.abnormal as c_abnormal
            |
            GO FROM $-.abnormal_cause_id OVER Abnormal_cause_2_solution YIELD DISTINCT $-.c_abnormal as abnormal, $-.abnormal_cause as abnormal_cause, properties($$).name as abnormal_solution
            '''
        abnormal_ngql = '''
                    USE {};
                   (MATCH(v0: software_upgrate)-[e0: software_upgrate_2_software_version_f]-(v1:software_version)
                    where v1.name == '{}'
                    RETURN id(v0) as vid
                    INTERSECT
                    MATCH(v0: software_upgrate)-[e0: software_upgrate_2_software_version]-(v1:software_version)
                    where v1.name == '{}'
                    RETURN id(v0) as vid)
                    | 
                    GO FROM $-.vid OVER software_upgrate_2_abnormal 
                    where properties($$).name in {}
                    YIELD dst(edge) as abnormal_id, properties($$).name as abnormal
                    | 
                    GO FROM $-.abnormal_id OVER abnormal_2_Abnormal_cause YIELD DISTINCT dst(edge) as abnormal_cause_id ,properties($$).name as abnormal_cause, $-.abnormal as c_abnormal
                    |
                    GO FROM $-.abnormal_cause_id OVER Abnormal_cause_2_solution YIELD DISTINCT $-.abnormal_cause as abnormal_cause, properties($$).name as abnormal_solution
                    '''
        if abnormal_cate_mention:
            res = await graph_connector.execute_json(abnormal_cate_ngql.format(space_name, softwareVersion[0], softwareVersion[1], abnormal_cate_mention))
        if abnormal_mention:
            res = await graph_connector.execute_json(abnormal_ngql.format(space_name, softwareVersion[0], softwareVersion[1], abnormal_mention))

        return json.loads(res)


    async def exceptionInfo_intent_search_function(self, graph_connector, space_name, **kwargs):
        """
        "异常信息(有具体异常现象)-分析原因、处理方法"这一意图对应的图谱查询逻辑。
        :param graph_connector: graph db connector
        :param space_name: graph db space name
        :param **kwargs：entity mention extracted in qurey text , e.g. [Mention(text,type,s,e),Mention(text2,type2,s2,e2)]
        :return: json result, 给出引起这个中间状态的异常的原因有哪几种及解决方案
        Examples:
        >>> ItMaintenanceGraphSearch.exceptionInfo_intent_search_function(client, 'nebula0526', Mention('Pod 处于 ImagePullBackOff 状态', 'abnormal', 1, 27))
        {'errors': [{'code': 0}], 'results': [{'spaceName': 'nebula0526', 'data': [{'meta': [None, None], 'row': ['从私有镜像仓库拉取镜像', '配置从私有仓库拉取镜像']}, {'meta': [None, None], 'row': ['镜像名错误', '修复镜像名']}, {'meta': [None, None], 'row': ['镜像无效、不存在', '修复tag']}], 'columns': ['abnormal_causes', 'method_disposal'], 'errors': {'code': 0}, 'latencyInUs': 13384}]}
        """
        abnormal_cate_mention = []
        res = {}
        for mention in kwargs.get('input_entities'):
            if mention.type == 'cate' or mention.type == 'abnormal':
                abnormal_cate_mention.append(mention.text)
        ngql = "USE {};" \
                "MATCH(v)-[e:abnormal_2_Abnormal_cause |:exception_cate_2_Abnormal_cause]-(v2: Abnormal_cause)-[e2: Abnormal_cause_2_solution]-(v3: solution)  " \
                "WHERE v.name in {} " \
                "RETURN v2.name AS abnormal_causes, v3.name AS method_disposal ;"
        res = await graph_connector.execute_json(ngql.format(space_name, abnormal_cate_mention))

        return json.loads(res)

    async def exceptionInfo_onlySoftware(self, graph_connector, space_name, **kwargs):
        """
        "异常信息(只有一个软件)-分析原因、处理方法"这一意图对应的图谱查询逻辑。
        :param graph_connector: graph db connector
        :param space_name: graph db space name
        :param **kwargs：entity mention extracted in qurey text , e.g. [Mention(text,type,s,e),Mention(text2,type2,s2,e2)]
        :return: json result, 给出引起这个中间状态的异常的原因有哪几种及解决方案
        Examples:
        >>> ItMaintenanceGraphSearch.exceptionInfo_intent_search_function(client, 'nebula0526', Mention('Pod 处于 ImagePullBackOff 状态', 'abnormal', 1, 27))
        {'errors': [{'code': 0}], 'results': [{'spaceName': 'nebula0526', 'data': [{'meta': [None, None], 'row': ['从私有镜像仓库拉取镜像', '配置从私有仓库拉取镜像']}, {'meta': [None, None], 'row': ['镜像名错误', '修复镜像名']}, {'meta': [None, None], 'row': ['镜像无效、不存在', '修复tag']}], 'columns': ['abnormal_causes', 'method_disposal'], 'errors': {'code': 0}, 'latencyInUs': 13384}]}
        """
        software = []
        abnormal = []
        supgrade = []
        exception_cate = []
        res = {}
        for mention in kwargs.get('input_entities'):
            if mention.type == 'software':
                software.append(mention.text)
            if mention.type == 'abnormal':
                abnormal.append(mention.text)
            if mention.type == 'supgrade_type':
                supgrade.append(mention.text)
            if mention.type == 'cate':
                exception_cate.append(mention.text)
        ngql1 = "USE {};" \
               "MATCH(v0: software)-[e0:software_2_software_upgrate]-(v1:software_upgrate)-[e1:software_upgrate_2_abnormal]-(v2: abnormal)-[e2: abnormal_2_Abnormal_cause]-(v3: Abnormal_cause)-[e3: Abnormal_cause_2_solution]-(v4: solution)  " \
               "WHERE v0.name in {} " \
               "RETURN v2.name as abnormal, v3.name AS abnormal_causes, v4.name AS method_disposal ;"
        ngql2 = "USE {};" \
               "MATCH(v0: software)-[e0:software_2_software_upgrate]-(v1:software_upgrate)-[e1:software_upgrate_2_abnormal]-(v2: abnormal)-[e2: abnormal_2_Abnormal_cause]-(v3: Abnormal_cause)-[e3: Abnormal_cause_2_solution]-(v4: solution)  " \
               "WHERE v0.name in {} and v2.name in {}" \
               "RETURN v2.name as abnormal, v3.name AS abnormal_causes, v4.name AS method_disposal ;"
        ngql3 = '''
            USE {};
            MATCH(v0: software)-[e0:software_2_abnormal]-(v2: abnormal)-[e2: abnormal_2_Abnormal_cause]-(v3: Abnormal_cause)-[e3: Abnormal_cause_2_solution]-(v4: solution)
            WHERE v0.name in {} and v2.name in {}
            RETURN v2.name as abnormal, v3.name AS abnormal_causes, v4.name AS method_disposal ;
        '''

        ngql4_abnormal = '''
            USE {};
            MATCH (v:software)-[e:software_2_abnormal]->(v0:abnormal)<-[e0:exception_cate_2_abnormal]-(v1:exception_cate)
            WHERE v.name in {} and v1.name in {} 
            RETURN v0.name as abnormal, '' AS abnormal_causes, '' AS method_disposal
        '''
        ngql4_abnormal_cause = '''
            USE {};
            MATCH (v:software)-[e:software_2_abnormal]->(v0:abnormal)<-[e0:exception_cate_2_abnormal]-(v1:exception_cate)
            WHERE v.name in {} and v1.name in {} 
            RETURN id(v0) as abnormal_id, v0.name as abnormal
            |
            GO FROM $-.abnormal_id OVER abnormal_2_Abnormal_cause YIELD DISTINCT properties($$).name as abnormal_cause, $-.abnormal as abnormal, '' AS method_disposal
        '''
        ngql4_all = '''
            USE {};
            MATCH (v:software)-[e:software_2_abnormal]->(v0:abnormal)<-[e0:exception_cate_2_abnormal]-(v1:exception_cate)
            WHERE v.name in {} and v1.name in {} 
            RETURN id(v0) as abnormal_id, v0.name as abnormal
            |
            GO FROM $-.abnormal_id OVER abnormal_2_Abnormal_cause YIELD DISTINCT dst(edge) as abnormal_cause_id ,properties($$).name as abnormal_cause, $-.abnormal as c_abnormal
            |
            GO FROM $-.abnormal_cause_id OVER Abnormal_cause_2_solution YIELD DISTINCT $-.c_abnormal as abnormal, $-.abnormal_cause as abnormal_cause, properties($$).name as abnormal_solution 
        '''
        if not abnormal and exception_cate:
            # res = await graph_connector.execute_json(ngql4.format(space_name, software, exception_cate))
            res_abnormal = json.loads(
                await graph_connector.execute_json(ngql4_abnormal.format(space_name, software, exception_cate)))
            res_abnormal_cause = json.loads(
                await graph_connector.execute_json(ngql4_abnormal_cause.format(space_name, software, exception_cate)))
            res_all = json.loads(await graph_connector.execute_json(ngql4_all.format(space_name, software, exception_cate)))
            if res_abnormal['errors'][0]['code']:
                return json.loads(res_abnormal)
            if res_abnormal_cause['errors'][0]['code']:
                return json.loads(res_abnormal_cause)
            if res_all['errors'][0]['code']:
                return json.loads(res_all)

            columns = res_abnormal["results"][0]["columns"]
            row_abnormal = []
            row_abnormal_cause = []
            row_all = []
            for meta in res_abnormal["results"][0]["data"]:
                row_abnormal.append(meta["row"])
            for meta in res_abnormal_cause["results"][0]["data"]:
                row_abnormal_cause.append(meta["row"])
            for meta in res_all["results"][0]["data"]:
                row_all.append(meta["row"])
            res_abnormal_df = pd.DataFrame(row_abnormal, columns=columns)
            res_abnormal_cause_df = pd.DataFrame(row_abnormal_cause, columns=columns)
            res_all_df = pd.DataFrame(row_all, columns=columns)
            df_merge = res_abnormal_cause_df.merge(res_all_df, how='outer')
            df_merge = df_merge.groupby(['abnormal', 'abnormal_causes'])['method_disposal'].sum().reset_index()
            data = []
            for idx, row in df_merge.iterrows():
                print(row)
                print(df_merge.iloc[idx].values.tolist())
                meta_data = {}
                meta_data["meta"] = [None, None, None]
                meta_data["row"] = df_merge.iloc[idx].values.tolist()
                data.append(meta_data)
            res = {
                'errors': [{'code': 0}],
                'results': [
                    {
                        'spaceName': space_name,
                        'data': data,
                        "columns": columns,
                        "errors": {"code": 0},
                        "latencyInUs": ''
                    }
                ]
            }
            return res
        elif abnormal and software and not supgrade:
            res = await graph_connector.execute_json(ngql3.format(space_name, software, abnormal))
        elif abnormal and software and supgrade:
            res = await graph_connector.execute_json(ngql2.format(space_name, software, abnormal))
        elif supgrade and software and not abnormal:
            res = await graph_connector.execute_json(ngql1.format(space_name, software))

        return json.loads(res)

    async def intent_search_softwareVersionException(self, graph_connector, space_name, **kwargs):
        """
        "软件版本异常-分析类型、原因、处理方法（只有一个软件版本）"这一意图对应的图谱查询逻辑。
        :param graph_connector: graph db connector
        :param space_name: graph db space name
        :param **kwargs：entity mention extracted in qurey text , e.g. [Mention(text,type,s,e),Mention(text2,type2,s2,e2)]
        :return: json result, 包括异常分类、异常分原因、处理方法
        Examples:
        >>> ItMaintenanceGraphSearch.intent_search_softwareVersionException(client, 'nebula0526', Mention('AS7019升级到AS7020主模块升级完成后proton_etcd起不来', 'abnormal', 1, 37))
        {'errors': [{'code': 0}], 'results': [{'spaceName': 'n0601', 'data': [{'meta': [None, None, None], 'row': ['Pod异常类别', 'etcd集群管理，“副本重启从集群中移除”行为不必要', '删除副本重启时从集群移除']}, {'meta': [None, None, None], 'row': ['etcd异常类别', 'etcd集群管理，“副本重启从集群中移除”行为不必要', '删除副本重启时从集群移除']}], 'columns': ['exception_cate', 'abnormal_causes', 'method_disposal'], 'errors': {'code': 0}, 'latencyInUs': 19959}]}
         """

        abnormal_mention = []
        for mention in kwargs.get('input_entities'):
            if mention.type == 'abnormal':
                abnormal_mention.append(mention.text)

        ngql = "USE {};" \
               "MATCH (v0: exception_cate)-[e0:exception_cate_2_abnormal]->(v: abnormal)-[e:abnormal_2_Abnormal_cause]->(v2: Abnormal_cause)-[e2: Abnormal_cause_2_solution]->(v3: solution) " \
               "WHERE v.name in {} " \
               "RETURN v0.name AS exception_cate, v2.name AS abnormal_causes, v3.name AS method_disposal "
        res = await graph_connector.execute_json(ngql.format(space_name, abnormal_mention))

        return json.loads(res)

    async def intent_search_UpgradeException_softwareVersion(self, graph_connector, space_name, **kwargs):
        """
        "升级异常-分析类型、原因、处理方法(有软件版本, 无异常情况具体信息）"这一意图对应的图谱查询逻辑。
        :param graph_connector: graph db connector
        :param space_name: graph db space name
        :param **kwargs：entity mention extracted in qurey text , e.g. [Mention(text,type,s,e),Mention(text2,type2,s2,e2)]
        :return: json result, 展示AS7019主模块升级报错topN列表：AS7019升级到AS7020主模块升级报错的几种常见类型及解决方案
        Examples:
        >>> ItMaintenanceGraphSearch.intent_search_UpgradeException_softwareVersion(client, 'nebula0526', Mention('7.0.1.9', 'software_version', 1, 6))
        {'errors': [{'code': 0}], 'results': [{'spaceName': 'k8s0516', 'data': [{'meta': [None, None, None], 'row': ['etcd异常类别', 'etcd集群管理，“副本重启从集群中移除”行为不必要', '删除副本重启时从集群移除']}], 'columns': ['exception_cate', 'abnormal_causes', 'method_disposal'], 'errors': {'code': 0}, 'latencyInUs': 17555}]}
        """

        software_version = []
        for mention in kwargs.get('input_entities'):
            if mention.type == 'software_version':
                software_version.append(mention.text)

        ngql = "USE {};" \
               "MATCH (v0: software_version)-[e0:software_upgrate_2_software_version_f]-(v1: software_upgrate)" \
               "-[e1:software_upgrate_2_abnormal]-(v2: abnormal)-[e2:abnormal_2_Abnormal_cause]->(v3: Abnormal_cause)" \
               "-[e3: Abnormal_cause_2_solution]->(v4: solution) " \
               "WHERE v0.name in {} " \
               "RETURN v2.name AS abnormal, v3.name AS abnormal_causes, v4.name AS method_disposal LIMIT 5 " \

        res = await graph_connector.execute_json(ngql.format(space_name, software_version))
        return json.loads(res)

    async def intent_search_UpgradeException_allType(self, graph_connector, space_name, **kwargs):
        """
        "升级异常 - 分析类型、原因、处理方法(无软件名称、无异常情况名称）"这一意图对应的图谱查询逻辑。
        :param graph_connector: graph db connector
        :param space_name: graph db space name
        :param **kwargs：entity mention extracted in qurey text , e.g. [Mention(text,type,s,e),Mention(text2,type2,s2,e2)]
        :return: json result, 展示升级报错的几种常见问题及解决方案
        Examples:
        >>> ItMaintenanceGraphSearch.intent_search_UpgradeException_allType(client, 'nebula0526')
        {'errors': [{'code': 0}], 'results': [{'spaceName': 'k8s0516', 'data': [{'meta': [None, None, None], 'row': ['etcd异常类别', 'etcd集群管理，“副本重启从集群中移除”行为不必要', '删除副本重启时从集群移除']}], 'columns': ['exception_cate', 'abnormal_causes', 'method_disposal'], 'errors': {'code': 0}, 'latencyInUs': 17555}]}
        """

        ngql = """
                USE {};
                MATCH (v0:software_upgrate)-[e0:software_upgrate_2_abnormal]->(v: abnormal)-[e:abnormal_2_Abnormal_cause]->(v2: Abnormal_cause)-[e2: Abnormal_cause_2_solution]->(v3: solution) 
                RETURN v.name AS abnormal, v2.name AS abnormal_causes, v3.name AS method_disposal  
               """
        res = await graph_connector.execute_json(ngql.format(space_name))
        return json.loads(res)

    async def intent_search_exceptionList(self, graph_connector, space_name, **kwargs):
        """
        "异常列表-软件、异常、分类列表"这一意图对应的图谱查询逻辑。
        :param graph_connector: graph db connector
        :param space_name: graph db space name
        :param **kwargs：entity mention extracted in qurey text , e.g. [Mention(text,type,s,e),Mention(text2,type2,s2,e2)]
        :return: json result, 展示软件异常现象有哪些、异常原因，对应的解决方案
        Examples:
        >>> ItMaintenanceGraphSearch.intent_search_exceptionList(client, 'nebula0526', Mention('AS', 'software', 1, 2))
        {'errors': [{'code': 0}], 'results': [{'spaceName': 'n0601', 'data': [{'meta': [None, None, None], 'row': ['升级过程策略引擎报错pod not ready', 'etcd pod 显示无法加入集群。 由于etcd pod部署为无状态服务， 三副本宕机、两副本会无法恢复', '替换内部etcd的proton-etcd']}, {'meta': [None, None, None], 'row': ['AS7019升级到AS7020主模块升级完成后proton_etcd起不来', 'etcd集群管理，“副本重启从集群中移除”行为不必要', '删除副本重启时从集群移除']}, {'meta': [None, None, None], 'row': ['AS7023升级到AS7024 mongodb升级有两个mongondb Pod Crash起不来', 'mongo 回滚失败导致pod  crash', '1、备份数据 2、重启pod']}], 'columns': ['abnormal', 'abnormal_causes', 'method_disposal'], 'errors': {'code': 0}, 'latencyInUs': 23136}]}
        """

        software_cate = []
        output_type = []
        for mention in kwargs.get('input_entities'):
            if mention.type == 'software' or mention.type == 'cate':
                software_cate.append(mention.text)
            if mention.type == 'because_type':
                output_type.append('because_type')
            if mention.type == 'abnormal_type':
                output_type.append('abnormal_type')
            if mention.type == 'resolve_type':
                output_type.append('resolve_type')
        '''
        USE {};
        MATCH (v0)-[e0:software_2_abnormal |:exception_cate_2_abnormal]-(v2: abnormal)
        -[e2:abnormal_2_Abnormal_cause]->(v3: Abnormal_cause)-[e3: Abnormal_cause_2_solution]->(v4: solution)
        WHERE v0.name in {} 
        RETURN id(v2) as abnormal_id, v2.name AS abnormal, v3.name AS abnormal_causes, v4.name AS method_disposal LIMIT 5 
        |
        GO FROM $-.abnormal_id OVER exception_cate_2_abnormal REVERSELY YIELD DISTINCT dst(edge) as abnormal_cause_id ,properties($$).name as exception_cate, $-.abnormal as abnormal
        '''

        ngql_abnormal = '''
                USE {};
                MATCH (v0)-[e0:software_2_abnormal |:exception_cate_2_abnormal]-(v2: abnormal)
                WHERE v0.name in {}  
                RETURN v2.name AS abnormal, '' AS abnormal_causes, '' AS method_disposal
                '''
        ngql_abnormal_cause = '''
                        USE {};
                        MATCH (v0)-[e0:software_2_abnormal |:exception_cate_2_abnormal]-(v2: abnormal)
                        -[e2:abnormal_2_Abnormal_cause]->(v3: Abnormal_cause)
                        WHERE v0.name in {}  
                        RETURN v2.name AS abnormal, v3.name AS abnormal_causes, '' AS method_disposal
                        '''
        ngql_all = '''
               USE {};
               MATCH (v0)-[e0:software_2_abnormal |:exception_cate_2_abnormal]-(v2: abnormal)
               -[e2:abnormal_2_Abnormal_cause]->(v3: Abnormal_cause)-[e3: Abnormal_cause_2_solution]->(v4: solution)
               WHERE v0.name in {} 
               RETURN v2.name AS abnormal, v3.name AS abnormal_causes, v4.name AS method_disposal  
               '''
        ngql_because_type = '''
                       USE {};
                       MATCH (v0)-[e0:software_2_abnormal |:exception_cate_2_abnormal]-(v2: abnormal)
                       -[e2:abnormal_2_Abnormal_cause]->(v3: Abnormal_cause)
                       WHERE v0.name in {} 
                       RETURN v3.name AS abnormal_causes 
                       '''
        if ['because_type'] == output_type:
            res = await graph_connector.execute_json(ngql_because_type.format(space_name, software_cate))
            return json.loads(res)
        else:
            res_abnormal = json.loads(await graph_connector.execute_json(ngql_abnormal.format(space_name, software_cate)))
            res_abnormal_cause = json.loads(await graph_connector.execute_json(ngql_abnormal_cause.format(space_name, software_cate)))
            res_all = json.loads(await graph_connector.execute_json(ngql_all.format(space_name, software_cate)))
            if res_abnormal['errors'][0]['code']:
                return json.loads(res_abnormal)
            if res_abnormal_cause['errors'][0]['code']:
                return json.loads(res_abnormal_cause)
            if res_all['errors'][0]['code']:
                return json.loads(res_all)

            columns = res_abnormal["results"][0]["columns"]
            row_abnormal = []
            row_abnormal_cause = []
            row_all = []
            for meta in res_abnormal["results"][0]["data"]:
                row_abnormal.append(meta["row"])
            for meta in res_abnormal_cause["results"][0]["data"]:
                row_abnormal_cause.append(meta["row"])
            for meta in res_all["results"][0]["data"]:
                row_all.append(meta["row"])
            res_abnormal_df = pd.DataFrame(row_abnormal, columns=columns)
            res_abnormal_cause_df = pd.DataFrame(row_abnormal_cause, columns=columns)
            res_all_df = pd.DataFrame(row_all, columns=columns)
            df_merge = res_abnormal_cause_df.merge(res_all_df, how='outer')
            df_merge = df_merge.groupby(['abnormal', 'abnormal_causes'])['method_disposal'].sum().reset_index()
            data = []
            for idx, row in df_merge.iterrows():
                print(row)
                print(df_merge.iloc[idx].values.tolist())
                meta_data = {}
                meta_data["meta"] = [None, None, None]
                meta_data["row"] = df_merge.iloc[idx].values.tolist()
                data.append(meta_data)
            res = {
                'errors': [{'code': 0}],
                'results': [
                    {
                        'spaceName': space_name,
                        'data': data,
                        "columns": columns,
                        "errors": {"code": 0},
                        "latencyInUs": ''
                    }
                ]
            }

            # df_merge = res_abnormal_df.merge(res_abnormal_cause_df, on='abnormal').merge(res_all_df, on='abnormal')
            # df_merge['abnormal_causes'] = df_merge['abnormal_causes'] + df_merge['abnormal_causes_x'] + df_merge['abnormal_causes_y']
            # df_merge['method_disposal'] = df_merge['method_disposal'] + df_merge['method_disposal_x'] + df_merge['method_disposal_y']
            # res = df_merge[['abnormal', 'abnormal_causes', 'method_disposal']]
            return res

    async def intent_search_commandQuery(self, graph_connector, space_name, **kwargs):
        """
        "命令查询-功能/排查异常/解决方案实体（名称匹配）->命令查询"这一意图对应的图谱查询逻辑。
        :param graph_connector: graph db connector
        :param space_name: graph db space name
        :param **kwargs：entity mention extracted in qurey text , e.g. [Mention(text,type,s,e),Mention(text2,type2,s2,e2)]
        :return: json result, 展示对应命令
        Examples:
        >>> ItMaintenanceGraphSearch.intent_search_commandQuery(client, 'nebula0526', Mention('获取列出一个或多个资源的信息', 'function', 1, 15), Mention('获取节点信息', 'command_option', 1, 7))
        {'errors': [{'code': 0}], 'results': [{'spaceName': 'n0601', 'data': [{'meta': [None, None, None], 'row': ['升级过程策略引擎报错pod not ready', 'etcd pod 显示无法加入集群。 由于etcd pod部署为无状态服务， 三副本宕机、两副本会无法恢复', '替换内部etcd的proton-etcd']}, {'meta': [None, None, None], 'row': ['AS7019升级到AS7020主模块升级完成后proton_etcd起不来', 'etcd集群管理，“副本重启从集群中移除”行为不必要', '删除副本重启时从集群移除']}, {'meta': [None, None, None], 'row': ['AS7023升级到AS7024 mongodb升级有两个mongondb Pod Crash起不来', 'mongo 回滚失败导致pod  crash', '1、备份数据 2、重启pod']}], 'columns': ['abnormal', 'abnormal_causes', 'method_disposal'], 'errors': {'code': 0}, 'latencyInUs': 23136}]}

        """

        software = []
        for mention in kwargs.get('input_entities'):
            if mention.type == 'software':
                software.append(mention.text)

        ngql = '''
               USE {}; 
               MATCH (v0:{})-[e0]->(v1:command)-[e]->(v2:command_option) 
               WHERE v0.name == "{}" AND properties(v2).`describe` == "{}"
               RETURN id(v1) as vid, v1.name as p_command, v2.name as command_option 
               |
               GO FROM $-.vid OVER command_2_command REVERSELY YIELD DISTINCT properties($$).name, $-.p_command, $-.command_option
               '''
        res = await graph_connector.execute_json(ngql.format(space_name, "function", "获取列出一个或多个资源的信息", "获取节点信息"))

        return json.loads(res)

    async def intent_search_upgradeList(self, graph_connector, space_name, **kwargs):
        """
        "软件版本实体->通过关系关联升级服务实体列表"这一意图对应的图谱查询逻辑。
        :param graph_connector: graph db connector
        :param space_name: graph db space name
        :param **kwargs：entity mention extracted in qurey text , e.g. [Mention(text,type,s,e),Mention(text2,type2,s2,e2)]
        :return: json result, 展示升级服务列表
        Examples:
        >>> ItMaintenanceGraphSearch.intent_search_upgradeList(client, 'nebula0526', Mention('7.0.1.9', 'software_version', 1, 8))
        {'errors': [{'code': 0}], 'results': [{'spaceName': 'n0601', 'data': [{'meta': [None], 'row': ['AS 7.0.1.9-AS 7.0.2.0']}], 'columns': ['software_upgrat'], 'errors': {'code': 0}, 'latencyInUs': 18206}]}

        """

        software_version = []
        for mention in kwargs.get('input_entities'):
            if mention.type == 'software_version':
                software_version.append(mention.text)

        ngql = "USE {};" \
               "MATCH (v0: software_version)-[e0:software_upgrate_2_software_version_f]-(v1: software_upgrate)" \
               "WHERE v0.name in {} " \
               "RETURN v1.name AS software_upgrat "
        res = await graph_connector.execute_json(ngql.format(space_name, software_version))

        return json.loads(res)

    async def intent_search_softwareVersionLatest(self, graph_connector, space_name, **kwargs):
        """
        "软件版本推理-最新"这一意图对应的图谱查询逻辑。
        :param graph_connector: graph db connector
        :param space_name: graph db space name
        :param **kwargs：entity mention extracted in qurey text , e.g. [Mention(text,type,s,e),Mention(text2,type2,s2,e2)]
        :return: json result, 展示最新版本号
        Examples:
        >>> ItMaintenanceGraphSearch.intent_search_softwareVersionLatest(client, 'nebula0526', Mention('AS', 'software', 1, 3))
        {'errors': [{'code': 0}], 'results': [{'spaceName': 'n0601', 'data': [{'meta': [None], 'row': ['7.0.3.6']}], 'columns': ['software_version'], 'errors': {'code': 0}, 'latencyInUs': 11655}]}

        """

        for mention in kwargs.get('input_entities'):
            if mention.type == 'software':
                software = mention.text

        ngql = 'USE {};' \
               'MATCH (v:software)-[e:software_2_software_version]->(v1:software_version) ' \
               'where properties(v).name == "{}" RETURN v1.name as software_version ' \
               'MINUS ' \
               'MATCH (v:software)-[e:software_2_software_version]->(v1:software_version)-[e1:software_version_2_software_version]->(v2:software_version) ' \
               'where properties(v).name == "{}" RETURN v1.name as software_version '
        res = await graph_connector.execute_json(ngql.format(space_name, software, software))

        return json.loads(res)

    async def intent_search_softwareVersionCount(self, graph_connector, space_name, **kwargs):
        """
        "软件版本推理-统计"这一意图对应的图谱查询逻辑。
        :param graph_connector: graph db connector
        :param space_name: graph db space name
        :param **kwargs：entity mention extracted in qurey text , e.g. [Mention(text,type,s,e),Mention(text2,type2,s2,e2)]
        :return: json result, 展示有几个版本
        Examples:
        >>> ItMaintenanceGraphSearch.intent_search_softwareVersionCount(client, 'nebula0526', Mention('AS', 'software', 1, 3))
        {'errors': [{'code': 0}], 'results': [{'spaceName': 'n0601', 'data': [{'meta': [None], 'row': [16]}], 'columns': ['count(v0.name)'], 'errors': {'code': 0}, 'latencyInUs': 8072}]}

        """

        for mention in kwargs.get('input_entities'):
            if mention.type == 'software':
                software = mention.text

        # 软件版本推理-统计
        ngql = "USE {};" \
               "MATCH p=(v:software)-[e:software_2_software_version]->(v0:software_version) " \
               "where properties(v).name == '{}' RETURN properties(v).name as software, count(v0.name) as number_versions"
        res = await graph_connector.execute_json(ngql.format(space_name, software))

        return json.loads(res)

    async def intent_search_UpdateProcedures(self, graph_connector, space_name, **kwargs):
        """
        "升级流程步骤"这一意图对应的图谱查询逻辑。
        :param graph_connector: graph db connector
        :param space_name: graph db space name
        :param **kwargs：entity mention extracted in qurey text , e.g. [Mention(text,type,s,e),Mention(text2,type2,s2,e2)]
        :return: json result, 展示升级步骤
        Examples:
        >>> ItMaintenanceGraphSearch.intent_search_UpdateProcedures(client, 'nebula0526', Mention('7.0.1.9', 'software_version', 1, 8), Mention('7.0.2.0', 'software_version', 1, 8))
        {'errors': [{'code': 0}], 'results': [{'spaceName': 'n0601', 'data': [{'meta': [None], 'row': [16]}], 'columns': ['count(v0.name)'], 'errors': {'code': 0}, 'latencyInUs': 8072}]}

        """

        software_version = []
        for mention in kwargs.get('input_entities'):
            if mention.type == 'software':
                software_version.append(mention.text)

        ngql = "USE {};" \
               "MATCH p=(v:software)-[e:software_2_software_version]->(v0:software_version) " \
               " "
        res = await graph_connector.execute_json(ngql.format(space_name, software_version))

        return json.loads(res)

    async def intent_search_concept(self, graph_connector, space_name, **kwargs):
        """
        "名词解释"这一意图对应的图谱查询逻辑。
        :param graph_connector: graph db connector
        :param space_name: graph db space name
        :param **kwargs：entity mention extracted in qurey text , e.g. [Mention(text,type,s,e),Mention(text2,type2,s2,e2)]
        :return: json result, 展示名词概念
        Examples:
        >>> ItMaintenanceGraphSearch.intent_search_concept(client, 'concept', Mention('虚拟化技术', 'concept', 0, 4))
        {'errors': [{'code': 0}], 'results': [{'spaceName': 'concept', 'data': [{'meta': [None], 'row': ['VT，就是虚拟化技术（Virtualization Technology）的缩写。Intel VT就是指Intel的虚拟化技术。这种技术简单来说就是可以让一个CPU工作起来就像多个CPU并行运行，从而使得在一台电脑内可以同时运行多个操作系统。只有部分Intel 的CPU才支持这种技术。']}], 'columns': ['v.define'], 'errors': {'code': 0}, 'latencyInUs': 7248}]}
        """
        concept = []
        for mention in kwargs.get('input_entities'):
            if mention.type == 'concept':
                concept.append(mention.text)
        ngql = '''
            USE {};
            MATCH (v:concept)
            where v.name in {} 
            RETURN v.define
        '''
        res = await graph_connector.execute_json(ngql.format(space_name, concept))
        return json.loads(res)


if __name__ == '__main__':
    """
    连接到nebula graph服务
    host:10.4.131.25:9669     用户名：root  密码：root
    """
    # 定义配置
    config = Config()
    config.max_connection_pool_size = 10
    # 初始化连接池
    connection_pool = ConnectionPool()
    # 连接到graph服务地址端⼝
    connection_pool.init([('10.4.131.25', 9669)], config)
    # 从连接池中获取会话
    client = connection_pool.get_session('root', 'root')
    # 使⽤图空间
    client.execute("USE n0601")
    ngql = '''
           MATCH (v0:{})-[e0]->(v1:command)-[e]->(v2:command_option) 
           WHERE v0.name == "{}" AND properties(v2).`describe` == "{}"
           RETURN id(v1) as vid, v1.name as p_command, v2.name as command_option 
           |
           GO FROM $-.vid OVER command_2_command REVERSELY YIELD DISTINCT properties($$).name, $-.p_command, $-.command_option
           '''
    ngql = ngql.format("function", "获取列出一个或多个资源的信息", "获取节点信息")
    res = client.execute_json(ngql)
    print(json.loads(res))
    res = client.execute(ngql)
    # 释放会话
    client.release()

it_graph_search = ItMaintenanceGraphSearch()
