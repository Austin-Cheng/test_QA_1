import os
import time
from dataclasses import dataclass

import requests

from app.constants import config as GConstants
from common.logger import logging


@dataclass
class AlgSearchConfig:
    """
    文档查询服务的配置
    """
    ALG_HOST: str = os.getenv('ALG_HOST')
    ALG_PORT: str = os.getenv('ALG_PORT')
    ALG_PATH: str = os.getenv('ALG_PATH')


class AlgDocumentSearchEngine:
    config_id = os.getenv('ALG_CONFIG_ID')
    alg_server_config = AlgSearchConfig()

    @classmethod
    def alg_server_path(cls):
        c = cls.alg_server_config
        return "http://{}:{}{}{}".format(c.ALG_HOST, c.ALG_PORT, c.ALG_PATH, cls.config_id)

    @classmethod
    def search_document(cls, **kwargs):
        """
        调用alg_server的服务，查询出文档
        """
        query = kwargs.get('query')
        offset = kwargs.get('offset', GConstants.DEFAULT_OFFSET)
        limit = kwargs.get('limit', GConstants.DEFAULT_LIMIT)

        params = {
            "query": query,
            'page': int(offset / limit) + 1,
            'size': limit,
            'timestamp': time.time()
        }
        logging.debug("request params:{}".format(repr(params)))
        resp = requests.get(cls.alg_server_path(), params=params)
        try:
            return cls.extract_docs_resp(resp.json(), **kwargs)
        except Exception as e:
            logging.info("extract result errors {}".format(e))
        return None

    @classmethod
    def extract_docs_resp(cls, resp, **kwargs):
        """
        整理一下返回值
        """
        if resp is None or 'res' not in resp or 'search' not in resp['res']:
            return None

        searchList = resp['res']['search']

        docs = []
        for search in searchList:
            doc = dict()
            doc['display'] = search['properties']
            # 是否需要图谱参数
            doc["graph_info"] = search['search_path'] if kwargs.get('graph') else None
            doc['link'] = doc['display']['link']
            docs.append(doc)

        return {
            "query": kwargs.get('query'),
            'total_count': resp.get('number', len(docs)),
            'search': {
                GConstants.DISPLAY_DOC_TYPE: docs
            }
        }
