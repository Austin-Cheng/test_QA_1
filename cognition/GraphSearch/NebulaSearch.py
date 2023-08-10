# -*- coding:utf-8 -*-


import logging
from typing import Iterable, List, Union, Any

from cognition.GraphSearch.GraphSearch import (
    GraphSearch,
    TagType, Statement, OperateEnum, Path, Vertex, Edge, ALL_VERTEX_TYPE,
    DIRECTION,
    EmptySearchException, InvalidTypeException
)


class Query:
    """
    build n_gql where statement
    """

    def __init__(self, tag_type: TagType, tag_name: str):
        self.tag_type: TagType = tag_type
        self.tag_name: str = tag_name
        self._statement: List[Statement] = []

    def _append(self, statement: Statement):
        self._statement.append(statement)

    @staticmethod
    def _check_property_name(property_name: str):
        if not property_name:
            raise EmptySearchException(f'property_name:{property_name}')
        if not isinstance(property_name, str):
            raise InvalidTypeException(f'property_name:{property_name}')

    def _base_operate(self, property_name: str, value: Any, operate: OperateEnum) -> 'Query':
        """
        base operate method
        :param property_name: property name
        :param value: str or int
        :param operate: ==/>/>=/</<=/in/not in
        :return:
        """
        self._check_property_name(property_name)
        self._append(Statement(property_name, value, operate))
        return self

    def eq(self, property_name: str, value: Union[str, int]) -> 'Query':
        """
        :param property_name:
        :param value:
        :return:
        """
        return self._base_operate(property_name, value, OperateEnum.EQ)

    def gt(self, property_name: str, value: Union[str, int]) -> 'Query':
        return self._base_operate(property_name, value, OperateEnum.GT)

    def gte(self, property_name: str, value: Union[str, int]) -> 'Query':
        return self._base_operate(property_name, value, OperateEnum.GTE)

    def lt(self, property_name: str, value: Union[str, int]) -> 'Query':
        return self._base_operate(property_name, value, OperateEnum.LT)

    def lte(self, property_name: str, value: Union[str, int]) -> 'Query':
        return self._base_operate(property_name, value, OperateEnum.LTE)

    def in_(self, property_name: str, value: Iterable[Union[str, int]]) -> 'Query':
        return self._base_operate(property_name, value, OperateEnum.IN)

    def not_in_(self, property_name: str, value: Iterable[Union[str, int]]) -> 'Query':
        return self._base_operate(property_name, value, OperateEnum.NOT_IN)

    def is_null(self, property_name: str) -> 'Query':
        return self._base_operate(property_name, None, OperateEnum.IS_NULL)

    def is_not_null(self, property_name: str) -> 'Query':
        return self._base_operate(property_name, None, OperateEnum.IS_NOT_NULL)

    def is_empty(self, property_name: str) -> 'Query':
        return self._base_operate(property_name, None, OperateEnum.IS_EMPTY)

    def is_not_empty(self, property_name: str) -> 'Query':
        return self._base_operate(property_name, None, OperateEnum.IS_NOT_EMPTY)

    def generate_lookup_statement(self) -> str:
        """
        LOOKUP ON player WHERE player.age == 40 YIELD vertex as v;
        :return:
        """

        return self.generate_lookup_statement_by_statements(self.tag_type, self.tag_name, self._statement)

    @staticmethod
    def generate_lookup_statement_by_statements(tag_type: TagType, tag_name: str, statements: List[Statement]) -> str:
        """
        LOOKUP ON player WHERE player.age == 40 YIELD vertex as v;
        :param tag_type:
        :param tag_name:
        :param statements:
        :return:
        """

        def deal_value(value: Union[str, int]) -> str:
            if isinstance(value, str):
                return f'"{value}"'
            if isinstance(value, int):
                return f'{value}'
            raise InvalidTypeException(f'value:{value}')

        statement: str = f'LOOKUP ON {tag_name} WHERE'
        for _s in statements:
            statement += f' {tag_name}.{_s.property} {_s.operate.value}'
            # no need value
            if _s.operate in (
                    OperateEnum.IS_NULL, OperateEnum.IS_NOT_NULL,
                    OperateEnum.IS_EMPTY, OperateEnum.IS_NOT_EMPTY
            ):
                statement += ' AND'
                continue

            _value = _s.value
            # deal with in / not in
            if _s.operate in (OperateEnum.IN, OperateEnum.NOT_IN):
                statement += ' ['
                if isinstance(_value, str) or isinstance(_value, int):
                    _value = [_value]

                # iter _value
                if isinstance(_value, Iterable):
                    for _i in _value:
                        statement += f' {deal_value(_i)},'
                    statement = statement.rstrip(',')
                statement += ']'
            else:
                if not isinstance(_s.value, int) and not isinstance(_s.value, str):
                    raise InvalidTypeException(f'value:{_s.value}')
                statement += f' {deal_value(_value)}'

            statement += ' AND'
        # remove the last AND
        statement = statement.rstrip(' AND')
        # add yield statement
        if tag_type == TagType.VERTEX:
            statement += ' YIELD properties(vertex) as v'
        if tag_type == TagType.EDGE:
            statement += ' YIELD properties(edge) as v'
        statement += ';'
        return statement


class NebulaSearch(GraphSearch):
    """
    GraphSearch中的方法在此都要实现一遍
    """

    def vid_search(
            self, graph_connector, space_name: str, *vid: Union[str, int], vertex_types: List[str]
    ) -> List[Vertex]:
        """
            search vertex by vid, and then format data
        :param graph_connector: graph db connector
        :param space_name: graph db space name
        :param vid: graph db vertex'id
        :param vertex_types: graph db vertex vertex type list
        :return: List[Vertex]
        Examples:
        # FETCH PROP ON player "player101", "player102", "player103";
        >>> search: GraphSearch = GraphSearch()
        >>> vertex_list: List[Vertex] = search.vid_search('player100', 'player101')
        """
        if not vid:
            raise EmptySearchException('empty vid')
        # define return variable
        vertex_list: List[Vertex] = list()
        # build tag component
        vertex_type_component: str = ALL_VERTEX_TYPE
        if vertex_types:
            vertex_type_component = ','.join(vertex_types)
        # build vid component
        vid_component: str = ''
        for _v in vid:
            if isinstance(_v, int):
                vid_component += f',{_v}'
                continue
            if isinstance(_v, str):
                vid_component += f',"{_v}"'
                continue
            raise InvalidTypeException(str(type(_v)))
        vid_component.lstrip(',')

        # build n_gql
        n_gql: str = f'FETCH PROP ON {vertex_type_component} {vid_component}'
        logging.debug(f"vid_search(vid: {vid}, vertex_types:{vertex_type_component}) --> {n_gql}")
        query_res: Any = self.execute(graph_connector, space_name, n_gql)
        # format query result
        return vertex_list

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
        >>> vertex_list: List[Vertex] = search.search_entities('player', statements)
        """
        # define return variable
        vertex_list: List[Vertex] = list()
        # call Query.generate_lookup_statement_by_statements to generate n_gql
        n_gql: str = Query.generate_lookup_statement_by_statements(TagType.VERTEX, vertex_type, statements)
        logging.debug(f"search_entities(vertex_type: {vertex_type}, statements:{statements}) --> {n_gql}")
        query_res: Any = self.execute(graph_connector, space_name, n_gql)
        # format query result
        return vertex_list

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
        >>> edge_list: List[Edge] = search.search_edges('follow', statements)
        """
        # define return variable
        edge_list: List[Edge] = list()
        # call Query.generate_lookup_statement_by_statements to generate n_gql
        n_gql: str = Query.generate_lookup_statement_by_statements(TagType.EDGE, edge_type, statements)
        logging.debug(f"search_entities(edge_type: {edge_type}, statements:{statements}) --> {n_gql}")
        query_res: Any = self.execute(graph_connector, space_name, n_gql)
        # format query result
        return edge_list

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
        >>> vertex_list: List[Vertex] = search.entitiy_neighbours('player', 'player100')
        """
        pass

    def deep_walk(
            self, graph_connector, space_name: str, vertex_type: str, vid: Union[str, int], edge_types: List[str]
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
        >>> path_list: List[Path] = search.deep_walk('player', 'player100')
        """
        pass

    @staticmethod
    def entities_to_subgraph(graph_connector, input_entities: Iterable, invalidate_edge_types: Iterable,
                             invalidate_target_entity_types: Iterable, max_path_lenth: int):
        # TODO 原琦，不仅开发，还需要写上英文注释。注意不同的graph，graph_type不同,搜索不同。通过一群实体搜索其最大长度内的所有相关实体形成的
        #  子图
        '''
        :param graph:
        :param input_entities:
        :param invalidate_edge_types:
        :param invalidate_target_entity_types:
        :param max_path_lenth:
        :return:
        '''
        return ["chemical10", "chemical11", "chemical12"]

    def execute(self, graph_connector, space_name: str, sql: str) -> Any:
        """
        Execute the sql statement directly
        :param graph_connector: graph db connector
        :param space_name: graph db space name
        :param sql: sql statement
        :return:
        """
        pass


if __name__ == '__main__':
    test_statement: str = Query.generate_lookup_statement_by_statements(
        TagType.EDGE,
        'player',
        [
            Statement(property='name', value='aaa', operate=OperateEnum.EQ),
            Statement(property='version', value=None, operate=OperateEnum.IS_NULL),
            Statement(property='age', value=40, operate=OperateEnum.GTE),
        ]
    )
    print(test_statement)

    query: Query = Query(TagType.EDGE, 'player').eq('name', 'aaa').is_null('version').gte('age', 40)
    print(query.generate_lookup_statement())
