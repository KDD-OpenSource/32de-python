from api.neo4j_own import Neo4j
import logging
from multiprocessing import Pool

extracted_domains = []
logger = logging.getLogger('MetaExp.LAST_NIGHT_BATCHWISE_EDGE_DELETE')

bolt_url_freebase = 'bolt://mattis.malmo.neohq.net:7717/'
bolt_url_local = 'bolt://10.188.123.112:7697/'

print("Extract Domains")

counter = 0
domain_batch = []

with open('last_night_hacking/UninterestingDomains.txt', 'r') as uninteresting_domains_file:
	for line in uninteresting_domains_file:
		counter += 1
		if counter > 500:
			counter = 1
			extracted_domains.append(domain_batch)
			domain_batch = []
		extracted_domain = line.replace('\n', '').replace('\r', '')
		domain_batch.append(extracted_domain)
	extracted_domains.append(domain_batch)

print(extracted_domains)


def run_query_for_domain(dom_batch):
	print("Start deleting domain batch {}".format(dom_batch[0]))
	with Neo4j(uri=bolt_url_freebase, user='neo4j', password='') as neo4j:
		print(neo4j.delete_edge_with_domain_batch(dom_batch))

	print("Finished deleting domain batch")


print("Start multiprocessing")

try:
	pool = Pool(35)
	pool.map(run_query_for_domain, extracted_domains)
finally:
	pool.close()
	pool.join()
