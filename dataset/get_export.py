#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Exports tabular data for analysis
__author__ = """Giovanni Colavizza"""

import numpy as np
import logging, codecs, csv
logging.basicConfig(filename='logs/parser_stats.log',filemode="w+",level=logging.INFO)
logger = logging.getLogger("Main")

# MongoDB
from pymongo import MongoClient
from configparser import ConfigParser
config_set = "localhost" # this is in localhost
config = ConfigParser(allow_no_value=False)
config.read("config/config.conf")
mongo_db = config.get(config_set, 'db-name')
mongo_user = config.get(config_set, 'username')
mongo_pwd = config.get(config_set, 'password')
mongo_auth = config.get(config_set, 'auth-db')
mongo_host = config.get(config_set, 'db-host')
mongo_port = config.get(config_set, 'db-port')
client = MongoClient(mongo_host)
db = client[mongo_db]
db.authenticate(mongo_user, mongo_pwd, source=mongo_auth)

collection_publications = db.publications_dev
collection_stats = db.stats_dev
collection_authors = db.authors_dev

# GLOBALS
journal_classification_file = "config/sm_journal_classification_106_1.csv" # journal classification from Science-Metrix
out_file = "exports/export.csv"
separator = ";"
text_delim = '"'

if __name__ == "__main__":

	all_records = dict()
	filtered = 0
	journal_classification = dict()

	# LOAD journal classification (Science-Metrix)
	with codecs.open(journal_classification_file, encoding="utf-8-sig") as f:
		reader = csv.DictReader(f, delimiter=",", quotechar='"')
		for row in reader:
			if row["issn"]:
				journal_classification[row["issn"]] = row
			if row["essn"]:
				journal_classification[row["essn"]] = row

	# LOAD data
	for record in collection_publications.find():
		if not (record["is_plos"] or record["is_bmc"]):
			continue

		# sanitize
		title = record["title"].replace('"','')
		title = " ".join(title.split())
		das = record["das"].replace('"', '')
		das = " ".join(das.split())

		# FILTERS
		filter = False
		# remove review and editorial articles
		all_kw_and_sbj = record["keywords"]
		all_kw_and_sbj.extend(record["subjects"])
		all_kw_and_sbj = list(set(all_kw_and_sbj))
		for kwsbj in all_kw_and_sbj:
			merged_knsbj = "".join(kwsbj.split())
			if merged_knsbj.find("review") >= 0 or merged_knsbj.find("editor") >= 0:
				filter = True
				break
		# find journal classification
		journal_domain = ""
		journal_field = ""
		journal_subfield = ""
		for issn in record["journal_issn"]:
			if issn in journal_classification.keys():
				journal_domain = journal_classification[issn]["Domain_English"]
				journal_field = journal_classification[issn]["Field_English"]
				journal_subfield = journal_classification[issn]["SubField_English"]
		if filter:
			filtered += 1
			continue

		all_records[record["_id"]] = [str(record["id_pmid"]), str(record["id_pmc"]), record["id_doi"], record["id_publisher"], record["journal"], journal_domain, journal_field, journal_subfield,
		                              str(record["n_authors"]), record["is_plos"], record["is_bmc"],
		                              title, str(record["n_references"]), str(len(record["references"])), record["has_das"], record["das_encouraged"], record["das_required"], das]


	for record in collection_stats.find():
		if not (record["is_plos"] or record["is_bmc"]):
			continue

		if not record["publication_id"] in all_records.keys():
			logger.warning("Issue with publication_id: "+str(record["publication_id"]))
			continue

		all_records[record["publication_id"]].extend(
			[str(record["year"]), str(record["month"]), record["has_month"], str(record["citations_two"]), str(record["citations_three"]), str(record["citations_five"]), str(record["citations_total"])])
		if record["h_indexes"]:
			all_records[record["publication_id"]].extend([str(np.min(record["h_indexes"])), str(np.max(record["h_indexes"])),
			                                              '%.3f'%np.mean(record["h_indexes"]), '%.3f'%np.median(record["h_indexes"])])
		else: # if h_indexes are missing
			all_records[record["publication_id"]].extend([None,None,None,None])

	# EXPORT main table
	with codecs.open(out_file,"w",encoding="utf8") as f:
		writer = csv.writer(f, delimiter=separator, quotechar=text_delim, quoting=csv.QUOTE_MINIMAL)
		writer.writerow(["pmid","pmcid","doi","publisher_id","journal","journal_domain","journal_field","journal_subfield","n_authors","is_plos","is_bmc","title","n_references_tot","n_references",
		                 "has_das","das_encouraged","das_required","das","p_year","p_month","has_month","n_cit_2","n_cit_3","n_cit_5","n_cit_tot","h_index_min","h_index_max","h_index_mean","h_index_median"])
		for k,v in all_records.items():
			writer.writerow(v)

	logger.info("Filtered: "+str(filtered))
	logger.info("All done!")