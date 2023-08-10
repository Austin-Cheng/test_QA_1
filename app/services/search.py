# -*- coding: utf-8 -*-

import inject
from fastapi import APIRouter, Query
from fastapi import HTTPException

from app.constants import config as GConstants
from app.handlers.alg_search import AlgDocumentSearchEngine
from app.handlers.model import BaseResponse
from app.handlers.search_engine import ItMaintenanceSearchEngine
from common.errorcode.codes import search as code
from common.errorcode.sview import Sview

router = APIRouter(prefix='/cognition')


@router.post('/search',
             tags=['搜索'],
             response_model=BaseResponse,
             response_model_exclude_none=True,
             summary="认知搜索接口",
             description="""基于Nebula2的认知搜索接口，运维知识服务的基础接口\n
    场景1 文本格式结果
       参数：query=pod故障有哪些
            dt=txt  
       特性：多个结果，默认分页返回
       
    场景2 带统计功能结果
       参数：query=AS版本有几个
            dt=txt
       
    场景3 文档格式结果
       参数：query=MySQL
            dt=doc
                        """)
async def search(query: str = Query(..., example='pod故障有哪些？', description="用户输入参数"),
                 dt: str = Query(example='txt', description="查询结果的类型, txt:文本形式，doc:文档形式", default='txt'),
                 graph: int = Query(example=0, description="查询结果是否需要带上图信息,0不带，1带", default=0),
                 offset: int = Query(example=0, description="分页偏移量", ge=0, default=GConstants.DEFAULT_OFFSET),
                 limit: int = Query(example=10, description="分页每页数量", gt=0, default=GConstants.DEFAULT_LIMIT,
                                    le=GConstants.MAX_OFFSET)):
    """
    注释文档
    """
    if query == "":
        raise HTTPException(status_code=400, detail="param [query] is empty")
    # 如果单独查询txt
    if dt == GConstants.DISPLAY_TXT_TYPE:
        engine: ItMaintenanceSearchEngine = inject.instance(ItMaintenanceSearchEngine)
        return await engine.search(query, offset=offset, limit=limit, graph=graph, dt=dt)
    # 如果单独查文档
    resp = AlgDocumentSearchEngine.search_document(query=query, offset=offset, limit=limit, graph=graph, dt=dt)
    if resp is None:
        return Sview.TErrorreturn(code.APP_HANDLERS_SERVICE_ALG_SEARCH_NO_RESULT)
    return Sview.json_return(resp)


@router.post('/concept',
             tags=['划词'],
             response_model=BaseResponse,
             response_model_exclude_none=True,
             summary="划词搜索接口",
             description="""基于Nebula2概念图谱的认知搜索，划词搜索接口\n
    特点如下：
        1. 默认给一个出纯文本结果，没有文档格式，不支持分页
        2. 输入的是概念名词，如果是句子，也只会识别其中的概念
        3. 支持给出图谱数据
             """)
async def search(query: str = Query(..., example='k8s？', description="输入的概念"),
                 graph: int = Query(example=0, description="查询结果是否需要带上图信息,0不带，1带", default=0)):
    if query == "":
        raise HTTPException(status_code=400, detail="param [query] is empty")
    engine: ItMaintenanceSearchEngine = inject.instance(ItMaintenanceSearchEngine)
    return await engine.concept(query, graph=graph)


@router.post('/recommend',
             tags=['推荐'],
             response_model=BaseResponse,
             response_model_exclude_none=True,
             summary="文档推荐接口",
             description="""基于Nebula2文档图谱的关联搜索，文档推荐接口\n
    特点如下：
        1. 默认给文档结果，不支持纯文本结果，支持分页
        2. 查询的是相关的推荐文档
        3. 支持给出图谱数据
        """)
async def search(query: str = Query(..., example='MySQL', description="用户输入问题"),
                 graph: int = Query(example=0, description="查询结果是否需要带上图信息,0不带，1带", default=0),
                 offset: int = Query(example=0, description="分页偏移量", default=0),
                 limit: int = Query(example=10, description="分页每页数量", default=10)):
    if query == "":
        raise HTTPException(status_code=400, detail="param [query] is empty")
    resp = AlgDocumentSearchEngine.search_document(query=query, offset=offset, limit=limit, graph=graph)
    if resp is None:
        return Sview.TErrorreturn(code.APP_HANDLERS_SERVICE_ALG_SEARCH_NO_RESULT)
    return Sview.json_return(resp)
