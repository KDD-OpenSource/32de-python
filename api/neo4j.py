from neo4j import GraphDatabase, Node
from typing import List


class Neo4j:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def start_precomputation(self, mode:str):
        # mode = {full, high-degree}
        pass

    def get_metapaths(self, nodeset_A: List[Node], nodeset_B: List[Node], length:int):
        pass

    def get_all_metapaths(self):
        pass

    # TODO: Implement different return types (node instances, node types)
    # TODO: What are the parameters of the neo4j procedure?
    def random_walk(self, maybe_start_id: int, maybe_number_of_random_walks: int, maybe_walk_length: int):
        with self._driver.session() as session:
            # TODO: Is the result returned as paths?
            maybe_paths = session.run(
                "CALL random_walk($maybe_start_id, $maybe_number_of_random_walks, $maybe_walk_length);",
                maybe_start_id=maybe_start_id, maybe_number_of_random_walks=maybe_number_of_random_walks,
                maybe_walk_length=maybe_walk_length)
