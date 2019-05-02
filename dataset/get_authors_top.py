#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Exports a CSV with top authors and their publications, to check a) disambiguation and b) h-index
__author__ = """Giovanni Colavizza"""

import codecs

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

collection = db.stats_dev
collection_authors = db.authors_dev

limit = 10

# get records
authors = list()
for record in collection_authors.find():
	authors.append({"tot_cit":record["tot_cit"],"name":record["name"],"n_pub":len(record["publications"]),"publications":record["publications"],"h_index":record["h_index"]})

authors = sorted(authors,key=lambda x:x["n_pub"],reverse=True)
with codecs.open("exports/evaluation_authors.csv","w",encoding="utf8") as f:
	f.write("id,name,h_index,tot_cit,publication_title,pub_year,n_cit\n")
	for n,a in enumerate(authors[:limit]):
		for p in a["publications"]:
			title = p["title"].replace('"','')
			title = title.replace('\n', ' ')
			title = title.replace('\r', ' ')
			f.write(str(n)+","+'"'+a["name"]+'"'+","+str(a["h_index"])+","+str(a["tot_cit"])+","+'"'+title+'"'+","+str(p["year"])+","+str(p["n_cit"])+"\n")
