GRAPH_TYPE = set(['tree', 'normal_graph'])
GRAPH_DB_TYPE = set(['orient', 'neo4j', 'nebula'])


class GraphContent(object):

    def __init__(self, graph_type, graph_db_type, graph=None):
        if graph_type in GRAPH_TYPE and graph_db_type in GRAPH_DB_TYPE:
            self.graph_type = graph_type
            self.graph_db_type = graph_db_type
        else:
            raise ValueError(
                '''graph_type or graph_db_type unknown'''
            )
        self.graph = graph
