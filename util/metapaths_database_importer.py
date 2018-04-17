from tokenize import String, Number
import redis
import json

from typing import Dict, List


class RedisImporter:

    def __init__(self, host: String="localhost", port: Number=6379, password: String=None):
        self.database = redis.Redis(host=host, port=port, password=password)

    def importFile(self, filePath):
        data = json.load(open(filePath, "r", encoding="utf8"))
        return self.importJson(data)

    def importJson(self, data):
            self.writePaths(data['metapaths'])
            self.writeMappings(data['mapping']['node_types'], data['mapping']['edge_types'])

    def writePaths(self, paths: "List[String]"):
            for path in paths:
                self.writePath(path)

    def writePath(self, path: "String"):
        pathAsList = path.split("|")
        startNode = pathAsList[0]
        endNode = pathAsList[-1]
        self.database.lpush(startNode + " " + endNode, path)

    def writeMappings(self, node_type_mapping: "Dict[Number, String]", edge_type_mapping: "Dict[Number, String]"):
            self.writeMapping("node_type", node_type_mapping)
            self.writeMapping("edge_type", edge_type_mapping)

    def writeMapping(self, name: "String", mapping: "Dict[Number, String]"):
        key = name + "_mapping"
        for field in mapping.keys():
            self.database.hset(key, field, mapping[field])
            self.database.hset("reverse_" + key, mapping[field], field)
