from api.neo4j_own import Neo4j
import logging
from multiprocessing import Pool

extracted_domains = []
logger = logging.getLogger('MetaExp.LAST_NIGHT_BATCHWISE_EDGE_DELETE')

bolt_url_freebase = 'bolt://mattis.malmo.neohq.net:7717/'
bolt_url_local = 'bolt://10.188.123.112:7697/'

print("Extract Domains")

with open('last_night_hacking/UninterestingDomains.txt', 'r') as uninteresting_domains_file:
	for line in uninteresting_domains_file:
		extracted_domain = line.replace('\n', '').replace('\r', '')
		extracted_domains.append(extracted_domain)


def run_query_for_domain(domain):
	print("Start deleting domain {}".format(domain))
	with Neo4j(uri=bolt_url_freebase, user='neo4j', password='') as neo4j:
		neo4j.delete_edge_with_domain(domain)

	print("Finshed deleting domain {}".format(domain))


print("Start multiprocessing")

try:
	pool = Pool(140)
	pool.map(run_query_for_domain, extracted_domains)
finally:
	pool.close()
	pool.join()
