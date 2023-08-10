# -*- coding: utf-8 -*-


import os
from dataclasses import dataclass, Field
from typing import Dict, Any, List

from dacite import from_dict
from dacite.dataclasses import get_fields
from inject import autoparams

from app.handlers.graph_search import ItMaintenanceGraphSearch
from app.handlers.search_engine import ItMaintenanceSearchEngine
from cognition.Connector import RequestNebula
from common.logger import logging


@dataclass
class Config:
    PORT: int = os.getenv('PORT', 8080)
    PROJECT_NAME: str = os.getenv('PROJECT_NAME', 'it_maintenance_search')
    PROJECT_PATH: str = os.getenv('PROJECT_PATH')
    NEBULA_IPS = os.getenv('NEBULA_IPS').split(',')
    NEBULA_PORTS = os.getenv('NEBULA_PORTS').split(',')
    NEBULA_USER = os.getenv('NEBULA_USER')
    NEBULA_PASSWORD = os.getenv('NEBULA_PASSWORD')


@autoparams()
def bind_config() -> Config:
    _fields: List[Field] = get_fields(Config)
    config_dict: Dict[str, Any] = dict()
    # load config
    logging.info("loading config success")
    _config = from_dict(Config, config_dict)
    return _config


@autoparams()
def init_search_engine(config: Config) -> ItMaintenanceSearchEngine:
    graph_connector = RequestNebula(config.NEBULA_IPS, config.NEBULA_PORTS, config.NEBULA_USER, config.NEBULA_PASSWORD)
    return ItMaintenanceSearchEngine(graph_connector)


@autoparams()
def init_graph_search(config: Config) -> ItMaintenanceGraphSearch:
    return ItMaintenanceGraphSearch()


def bind(binder):
    """
    bind instance to inject container like spring
    we can get instance like below code
    >>> import inject
    >>> nebula: NebulaDb = inject.instance(NebulaDb)
    now we get nebula instance init by init_nebula_db function
    :param binder:
    :return:
    """
    # 初始化配置
    binder.bind_to_constructor(Config, bind_config)
    binder.bind_to_constructor(ItMaintenanceSearchEngine, init_search_engine)
    # 初始化主库
