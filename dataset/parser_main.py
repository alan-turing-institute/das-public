#!/usr/bin/python
# -*- coding: UTF-8 -*-
# PARSER main script of PubMed xml files
# Calls parser_function to parse every given list of PubMed xml files
__author__ = """Giovanni Colavizza"""

from parser_function import *

from collections import defaultdict
import pandas as pd
import logging, os

# logs and basics
logging.basicConfig(filename='logs/parser_main.log',filemode="w+",level=logging.INFO)
logger = logging.getLogger("Main")
list_of_journals_file = "config/journal_list.csv"
JOB_LIMIT = 50000 # controls how many articles to process per batch

# MongoDB
from pymongo import MongoClient
from pymongo import HASHED, ASCENDING
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

# 0 if Test, 1 if Production
MODE = int(config.get(config_set, 'mode'))

# select which collection to use in Mongo, start by dropping if needed (we do not update an existing collection here)
db.drop_collection("publications_dev")
collection = db.publications_dev

if __name__ == "__main__":

    root_dirs = ["dev_set"] # if you want to use a small, dev set sampled using sample_dev_set.py
    if MODE:
        # change this folder to where your PubMed OA dump actually is. You can list multiple folders, all xml files within will be processed
        root_dirs = ["PubMed/comm_use","PubMed/non_comm_use"]

    count_ids = defaultdict(int)
    logger.info("\n------------ \nNew JOB starting! \n")

    # prepare list of journals
    journals = pd.read_csv(list_of_journals_file)
    journals.dropna(subset=["folder_name"], inplace=True, how="all")
    journals.set_index("folder_name", inplace=True)
    journals.das_encouraged = pd.to_datetime(journals.das_encouraged).dt.date
    journals.das_required = pd.to_datetime(journals.das_required).dt.date
    # is_plos or bmc
    journals["is_plos"] = journals["product_title"].apply(lambda x:x.lower().startswith("plos "))
    list_of_journals = journals[["das_encouraged", "das_required", "is_plos"]].to_dict('index')

    # main parsing routine
    for root_dir in root_dirs:
        filenames = list()
        for root, dirs, files in os.walk(root_dir):
            filenames.extend([os.path.join(root,x) for x in files if "xml" in x])

        # drop duplicates
        all_filenames = len(filenames)
        filenames = list(set(filenames))
        logger.info("Not unique files: %d"%(all_filenames-len(filenames)))

        while len(filenames) > 0:  # we keep doing batches of work until there are no more new files per root folder

            local_filenames = filenames[:JOB_LIMIT] # add a job limit, if wanted. Run multiple times to finish it
            filenames = filenames[JOB_LIMIT:]
            new_files = len(local_filenames)
            logger.info("Processing %d files \n"%(len(local_filenames)))

            if len(local_filenames) > 0:
                # all the work is done here
                results = mp_article_parser(local_filenames, list_of_journals)
                logger.info("Finished parsing %d files"%len(local_filenames))

                # dump
                try:
                    if len(results) == 0:
                        logger.info("Empty result list! \n")
                    else:
                        collection.insert_many(results)
                except Exception as e:
                    if hasattr(e, 'message'):
                        logger.warning("Error in ingestion: %s \n" % e.message)
                    else:
                        logger.warning("Error in ingestion %s \n" % str(e))

                # count publication ids
                for r in results:
                    for key in r["identifiers"]:
                        count_ids[key["type"]] += 1

                del results

    count_ids = sorted({k: v for k, v in count_ids.items()}, key=lambda x: x[1], reverse=True)
    logger.info(str(count_ids))
    print(count_ids)

    # add indexes
    collection.create_index([('id_doi', HASHED)], background=True)
    collection.create_index([('id_pmc', ASCENDING)], background=True)
    collection.create_index([('id_pmid', ASCENDING)], background=True)

    logger.info("\nFinished!")
    print("\nFinished!")