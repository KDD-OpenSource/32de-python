from api.neo4j_own import Neo4j
import logging


logger = logging.getLogger('MetaExp.LAST_NIGHT_BATCHWISE_EDGE_DELETE')

bolt_url_freebase = 'bolt://mattis.malmo.neohq.net:7717/'
bolt_url_local = 'bolt://10.188.123.112:7697/'

with open('last_night_hacking/UninterestingDomains.txt', 'r') as uninteresting_domains_file:
	for line in uninteresting_domains_file:
		extracted_domain = line.replace('\n', '').replace('\r', '')
		logger.debug("EXTRACT DOMAIN {}".format(extracted_domain))
		with Neo4j(uri=bolt_url_freebase, user='neo4j', password='') as neo4j:
			neo4j.delete_edge_with_domain(extracted_domain)
