from neo4j.v1 import GraphDatabase, basic_auth

driver = GraphDatabase.driver("bolt://mattis.malmo.neohq.net:7717", auth=basic_auth("neo4j", "neo4j"))


session = driver.session()

with open("fuckDaShit.txt", "r") as file: 
	session.run(file.nextLine())

session.close()
