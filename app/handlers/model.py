import json
from typing import Optional

from pydantic import BaseModel, Field

from app.constants import config as GConstants
from app.constants.config import transDict
from app.utils.util import generate_random_str

class GraphEntity(BaseModel):
    edges: Optional[list] = Field(description="图的边信息")
    vertexes: Optional[list[dict]] = Field(description="查询结果相关实体信息")
    meta_data: Optional[list] = Field(description="图的元信息")

class SearchObject(BaseModel):
    display: dict = Field(description="查询结果，前端核心数据")
    link: Optional[str] = Field(description="查询结果的详情跳转链接")
    graph_info: Optional[GraphEntity] = Field(description="图谱实体信息")


class SearchModel(BaseModel):
    txt: Optional[list[SearchObject]] = Field(description="文本显示结果")
    doc: Optional[list[SearchObject]] = Field(description="文档显示结果")


class SearchResult(BaseModel):
    query: Optional[str] = Field(description="用户输入的查询语句")
    total_count: Optional[int] = Field(description="查询结果总数")
    search: Optional[SearchModel] = Field(description="查询结果")


class BaseResponse(BaseModel):
    error_code: Optional[str] = Field(description="错误码，按公司标准，改为字符串，正确情况下不返回")
    result: Optional[SearchResult] = Field(description="查询结果")
    description: Optional[str] = Field(description="错误，结果描述")


class SearchResultHelper():
    # 处理
    """
    处理： "results": [
              {
                "spaceName": "nebula",
                "data": [
                  {
                    "meta": [
                      null,
                      null,
                      null
                    ],
                    "row": [
                      "Pod异常类别",
                      "Pod 描述（例如你本地机器上的 mypod.yaml）中有问题",
                      "1.删除你的 Pod，并尝试带有 --validate 选项重新创建 2.检查的是 API 服务器上的 Pod 与你所期望创建的是否匹配"
                    ]
                  }
                ],
                "columns": [
                  "exception_cate",
                  "abnormal_causes",
                  "method_disposal"
                ],
                "errors": {
                  "code": 0
                },
                "latencyInUs": 10745
              }
            ]
    将其变为：
        {
         "spaceName": "nebula",
         "data":{
            "exception_cate":  "Pod异常类别",
            "abnormal_causes": "Pod 描述（例如你本地机器上的 mypod.yaml）中有问题",
            "method_disposal": "1.删除你的 Pod，并尝试带有 --validate 选项重新创建 2.检查的是 API 服务器上的 Pod 与你所期望创建的是否匹配"
          }
        }
    其余情况皆为异常情况
    """

    def filteIntentData(intentData):
        if not isinstance(intentData, list) and len(intentData) <= 0:
            return {}

        coloumns = dict(intentData[0]).get('columns', [])
        dataObj = dict(intentData[0]).get('data', '')

        datas = []
        keysDict = {index: key for index, key in enumerate(coloumns)}
        for solution in dataObj:
            rows = dict(solution).get('row')
            item = {}
            for k, v in keysDict.items():
                v = transDict.get(v, v)
                item[v] = rows[k]
            datas.append({"display": item})
        """如果没有合适的结果，返回空"""
        if len(datas) <= 0:
            return {}

        return {
            'txt': datas,
            'length': len(datas)
        }

    # 格式化结果，如果有一个意图成功，就视为成功
    @classmethod
    def formatSearchResult(cls, res, offset, limit, graph, dt):
        # 假分页处理
        if 'result' in res:
            result = res['result']
            if 'search' in result:
                search = result['search']
                txt_type = GConstants.DISPLAY_TXT_TYPE
                if len(search.get('txt', [])) > 0:
                    txtList = search[txt_type]
                    result['total_count'] = len(txtList)
                    search[txt_type] = txtList[offset: offset + limit]
                result['search'] = search
                res['result'] = result
        # 处理是否携带图信息
        if graph:
            res = cls.addGraph(res)
        # 有数据返回的
        if len(res.get('result', [])) > 0:
            if 'errorcode' in res and res.get('errorcode') == 0:
                res.pop('errorcode')
            if 'description' in res and res.get('description') == '':
                res.pop('description')
            return res
        # 结果为空处理
        res['errorcode'] = -1
        res['description'] = 'no search results'
        return res

    @classmethod
    def addQuery(cls, res, query: str):
        """
        添加query字段
        """
        if 'result' in res:
            result = res['result']
            result['query'] = query

    @classmethod
    def parseAndAppend(cls, res, data) -> int:
        """
        将部分结果解析，然后追加到最终结果后面，函数返回每次成功追加的数量
        res： 最终结果
        parts：单个意图的查询结果
        """
        if res is None or 'errors' not in data:
            return 0

        intentErrors = data['errors']
        if len(intentErrors) <= 0:
            return 0

        intentResult = {}
        # 1、处理错误
        intentResult['errorcode'] = dict(intentErrors[0]).get('code', 0)
        intentResult['message'] = dict(intentErrors[0]).get('message', '')
        # 1.1、如果报错了，那就返回 0，并且保留错误
        if intentResult.get('errorcode') != 0 and 'errorcode' not in res:
            res['errorcode'] = intentResult.get('errorcode')
            res['description'] = intentResult.get('message')
            return 0
        if intentResult.get('errorcode') != 0 and 'errorcode' in res:
            return 0
        # 2、处理results里面的数据
        if 'results' in data and isinstance(data['results'], list):
            intentData = cls.filteIntentData(data['results'])
            intentDataLength = intentData.get('length', 0)
            if intentData and intentDataLength > 0:
                if 'result' not in res:
                    res['result'] = {}
                results = res['result']
                search = results.get('search', {})
                # 2.1、初始化res结构
                txt_type = GConstants.DISPLAY_TXT_TYPE
                if txt_type in intentData and txt_type not in search:
                    search[GConstants.DISPLAY_TXT_TYPE] = []
                # 2.2、合并查询结果
                if txt_type in search and txt_type in intentData:
                    search[txt_type].extend(intentData[txt_type])
                results['search'] = search
                res['result'] = results
        if 'errorcode' not in res and intentDataLength > 0:
            res['errorcode'] = 0
        return intentDataLength

    @classmethod
    def graphJson(cls):
        return {
            "vertexes": [
                {
                    "id": "7a396f4f6f69b7cfb7891ea46c33db14",
                    "tag": "document",
                    "name": "多个软件提取MySQL、ORACLE、SqlServer",
                    "color": "#50A06A"
                }
            ],
            "edges": [],
            "meta_data": []
        }

    @classmethod
    def addGraph(cls, res):
        """
        生成假数据，图谱数据
        """
        if 'result' not in res:
            return
        result = res['result']
        if 'search' not in result:
            return
        search = result['search']

        if 'txt' in search:
            txtList = search['txt']
            for item in txtList:
                item['graph_info'] = cls.graphJson()
            search['txt'] = txtList

        if 'doc' in search:
            docList = search['doc']
            for item in docList:
                item['graph_info'] = cls.graphJson()
            search['doc'] = docList
        result['search'] = search
        res['result'] = result
        return res

    @classmethod
    def genConcept(cls):
        jsonStr = """{
                    "result": {
                        "search": {
                            "txt": [
                                {
                                    "display": {
                                        "简介": "升级过程策略引擎报错pod not ready",
                                        "特点": "etcd pod 显示无法加入集群。 由于etcd pod部署为无状态服务， 三副本宕机、两副本会无法恢复",
                                        "组件": "替换内部etcd的proton-etcd"
                                    },
                                    "link": "https://baike.baidu.com/item/kubernetes/22864162?fr=aladdin"
                                }
                            ]
                        }
                    }
                }"""
        return json.loads(jsonStr)
