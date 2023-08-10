# -*- coding: utf-8 -*-


import logging
from typing import Dict, Iterable, Union, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from cognition.GraphContent import GraphContent

# define const variable

#: all vertex type
ALL_VERTEX_TYPE = '*'


@dataclass
class Vertex:
    """
    describe a vertex
    """
    vid: Union[str, int]
    vertex_type: str
    # vertex props mapping
    props: Optional[Dict[str, Any]]


@dataclass
class Edge:
    """
    describe a edge
    """
    edge_type: str
    # edge props mapping
    props: Dict[str, Any]
    source: Vertex
    destination: Vertex


@dataclass
class Path:
    value: Vertex
    edge_type: str
    # edge props mapping
    props: Dict[str, Any]
    next: 'Path'


class TagType(str, Enum):
    VERTEX = 'VERTEX'
    EDGE = 'EDGE'


class DIRECTION(str, Enum):
    POSITIVE = 'POSITIVE'
    REVERSELY = 'REVERSELY'
    BIDIRECT = 'BIDIRECT'


class OperateEnum(str, Enum):
    EQ = "=="
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    IN = "IN"
    NOT_IN = "NOT IN"
    IS_NULL = 'IS NULL'
    IS_NOT_NULL = 'IS NOT NULL'
    IS_EMPTY = 'IS EMPTY'
    IS_NOT_EMPTY = 'IS NOT EMPTY'


@dataclass
class Statement:
    property: str
    value: Any
    operate: OperateEnum

    def __post_init__(self):
        # check value is iterable
        if self.operate in (OperateEnum.IN, OperateEnum.NOT_IN):
            if not isinstance(self.value, Iterable):
                raise InvalidTypeException(f'value:{self.value}')


class EngineSDKException(Exception):
    """
    search engine sdk base exception
    """

    def __init__(self, message: str = ''):
        super().__init__()

        self.message = message


class EmptySearchException(EngineSDKException):
    """
    Search for empty condition exception
    """
    pass


class InvalidTypeException(EngineSDKException):
    """
    param type invalid exception
    """
    pass


class GraphSearch:

    def vid_search(
            self, graph_connector, space_name: str, vertex_types: Optional[List[str]], *vid: Union[str, int]
    ) -> List[Vertex]:
        """
            search vertex by vid, and then format data
        :param graph_connector: graph db connector
        :param space_name: graph db space name
        :param vertex_types: graph db vertex vertex type list
        :param vid: graph db vertex'id
        :return: List[Vertex]
        Examples:
        # FETCH PROP ON player "player101", "player102", "player103";
        >>> search: GraphSearch = GraphSearch()
        >>> vertex_list: List[Vertex] = search.vid_search(conn, 'test', ['player'], 'player100', 'player101')
        """
        raise NotImplementedError

    def search_entities(
            self, graph_connector, space_name: str, vertex_type: str, statements: List[Statement]
    ) -> List[Vertex]:
        """
        search vertexs by vertex type and some statements
        :param graph_connector: graph db connector
        :param space_name: graph db space name
        :param vertex_type: vertex type
        :param statements: vertex properties statements
        :return: List[Vertex]
        Examples:
        # LOOKUP ON player WHERE player.age == 40 YIELD vertex as v;
        >>> search: GraphSearch = GraphSearch()
        >>> statements: List[Statement] = [Statement(property='name', value='aaa', operate=OperateEnum.EQ)]
        >>> vertex_list: List[Vertex] = search.search_entities(conn, 'test', 'player', statements)
        """
        raise NotImplementedError

    def search_edges(
            self, graph_connector, space_name: str, edge_type: str, statements: List[Statement]
    ) -> List[Edge]:
        """
        search edges by edge type and some statements
        :param graph_connector: graph db connector
        :param space_name: graph db space name
        :param edge_type: edge type
        :param statements: edge properties statements
        :return: List[Edge]
        Examples:
        # LOOKUP ON follow WHERE follow.degree == 90 YIELD edge as v;
        >>> search: GraphSearch = GraphSearch()
        >>> statements: List[Statement] = [Statement(property='degree', value=40, operate=OperateEnum.GT)]
        >>> edge_list: List[Edge] = search.search_edges(conn, 'test', 'follow', statements)
        """
        raise NotImplementedError

    def entitiy_neighbours(
            self, graph_connector, space_name: str, vertex_type: str, vid: Union[str, int],
            direction: DIRECTION = DIRECTION.POSITIVE
    ) -> List[Vertex]:
        """
        from entity search next steps entities
        :param graph_connector: graph db connector
        :param space_name: graph db space name
        :param vertex_type: vertex type
        :param vid: vertex vid
        :param direction: the direction of edge
        :return: List[Vertex]
        Examples:
        # MATCH (v:player)-[e*1]->(v2) WHERE id(v) == "player101" RETURN v2;
        >>> search: GraphSearch = GraphSearch()
        >>> vertex_list: List[Vertex] = search.entitiy_neighbours(conn, 'test', 'player', 'player100')
        """
        raise NotImplementedError

    def deep_walk(
            self, graph_connector, space_name: str, vertex_type: str, vid: Union[str, int],
            edge_types: Optional[List[str]]
    ) -> List[Path]:
        """
        from specific vertex walk through specific edges
        :param graph_connector: graph db connector
        :param space_name: graph db space name
        :param vertex_type: vertex type
        :param vid: vertex vid
        :param edge_types: edge type list
        :return: List[Path]
        Examples:
        MATCH p=(v:player)-[e]->(v2) WHERE id(v) == "player101" RETURN p;
        >>> search: GraphSearch = GraphSearch()
        >>> path_list: List[Path] = search.deep_walk(conn, 'test', 'player', 'player100', None)
        """
        raise NotImplementedError

    def entities_to_entities(
            self, graph_connector, space_name: str, input_entities: List[Vertex],
            invalidate_edge_types: List[str],
            invalidate_target_entity_types: List[str], max_path_lenth: int
    ) -> List[Vertex]:
        '''
        Search the relative target entities from input_entities through the edges not in the invalidate_edge_types and
        return the target entities who's type is not in invalidate_target_entity_types.
        :param graph_connector: graph db connector
        :param space_name: graph db space name
        :param input_entities:
        :param invalidate_edge_types:
        :param invalidate_target_entity_types:
        :param max_path_lenth:
        :return:
        '''
        pass

    def entities_to_subgraph(self, graph_connector, space, input_entities: Iterable,
                             invalidate_edge_types: Iterable,
                             invalidate_target_entity_types: Iterable, max_path_lenth: int):
        # TODO 原琦，不仅开发，还需要写上英文注释。注意不同的graph，graph_type不同,搜索不同。通过一群实体搜索其最大长度内的所有相关实体形成的
        #  子图
        '''
        :param graph_connector:
        :param space:
        :param input_entities:
        :param invalidate_edge_types:
        :param invalidate_target_entity_types:
        :param max_path_lenth:
        :return:
        '''
        raise NotImplementedError

    def find_path(self, graph_connector, space, start_entity: Vertex, end_entity: Vertex,
                  path_edge_type: str,
                  path_direction: DIRECTION, max_path_lenth: int, path_pattern: str) -> List[Path]:
        # TODO 原琦，不仅开发，还需要写上英文注释。注意不同的graph，graph_type不同,搜索不同。寻找路径。
        '''

        :param graph_connector:
        :param space:
        :param start_entity:
        :param end_entity:
        :param path_edge_type:
        :param path_direction:
        :param max_path_lenth:
        :param path_pattern:
        :return:
        '''
        raise NotImplementedError

    def execute(self, graph_connector, space_name: str, sql: str) -> Any:
        """
        Execute the sql statement directly
        :param graph_connector: graph db connector
        :param space_name: graph db space name
        :param sql: sql statement
        :return: connector returned original result
        Examples:
        MATCH p=(v:player)-[e]->(v2) WHERE id(v) == "player101" RETURN p;
        >>> search: GraphSearch = GraphSearch()
        >>> sql: str = 'MATCH p=(v:player)-[e]->(v2) WHERE id(v) == "player101" RETURN p;'
        >>> query_result: Any = search.execute(conn, 'test', sql)
        """
        pass
