#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Runs an evaluation of the data in the DB, using evaluation_plos.csv as input
__author__ = """Giovanni Colavizza"""

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

# load the eval csv, calculate from DB how many have DAS and not
eval_file = "config/evaluation_plos.csv"

eval_data = dict()
with open(eval_file) as f:
	for line in f:
		pm_id,tag = line.split(",")
		pm_id = int(pm_id.split("/")[-1][3:])
		has_das = True
		if "Y" in tag: # False positives are marked as Y, thus they have no das
			has_das = False
		eval_data[pm_id] = has_das

correct = 0
total = 0
missed = list()
# connect to db and get the records
for res in collection.find({"id_pmc": {"$in":[x for x in eval_data.keys()]}}):
	total += 1
	if eval_data[res["id_pmc"]] == res["has_das"]:
		correct += 1
	else:
		missed.append(res["id_pmc"])

if total > 0:
	print("Eval: ",correct/total)
	for miss in missed:
		print(miss)
		print(eval_data[miss])
else:
	print("No matches in this collection.")