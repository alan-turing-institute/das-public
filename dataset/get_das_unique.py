#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Exports a CSV with unique DAS, sorted by their counting and, eventually, journal. This is used for DAS classification
__author__ = """Giovanni Colavizza"""

import codecs
from collections import defaultdict, OrderedDict

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

collection = db.publications_dev

index_by_journal = dict()
index_global = defaultdict(int)

for record in collection.find():
	if record["has_das"]:
		internal_das = record["das"].replace('"', '')
		internal_das = ' '.join(internal_das.split())
		index_global[internal_das] += 1
		if record["journal"] not in index_by_journal.keys():
			index_by_journal[record["journal"]] = defaultdict(int)
		index_by_journal[record["journal"]][internal_das] += 1

# sorting
index_global = OrderedDict(sorted(index_global.items(), key=lambda x:x[1], reverse=True))
index_by_journal_list = sorted([(k,v,j) for j,d in index_by_journal.items() for k,v in d.items()], key=lambda x:x[1], reverse=True)

with codecs.open("das_classifier/input/das_full.csv", "w") as f:
	for k,v in index_global.items():
		f.write('"'+k+'",'+str(v)+"\n")

with codecs.open("das_classifier/input/das_journal.csv", "w") as f:
	for k in index_by_journal_list:
		f.write('"'+k[0]+'",'+str(k[1])+","+k[2]+"\n")