#!/usr/bin/python
# -*- coding: UTF-8 -*-
# SAMPLES from full PubMed collection for dev purposed
__author__ = """Giovanni Colavizza"""

import os, random, shutil

n_to_sample = 1000 # how many files to keep
root_folder = "ADD_YOURS_HERE" # root of PMC
destination = "dev_set" # folder where to store dev set

# cleanup and make fresh dev_set
os.removedirs(destination)
os.makedirs(destination)

# get all files
filenames = list()
for root, dirs, files in os.walk(root_folder):
	for f in files:
		if "xml" in f:
			filenames.append(os.path.join(root,f))

print(len(filenames))

# sample from list
selected = random.sample(filenames,n_to_sample)

# write them out
for f in selected:
	new_f = f.replace(root_folder,destination)
	os.makedirs(os.path.dirname(new_f),exist_ok=True)
	shutil.copyfile(f,new_f)

print("All done!")
