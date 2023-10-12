# Dataset

Folder containing the necessary code to create a dataset for analysis from the PubMed Central Open Access collection.

## Contents

* [config folder](config): contains config files, ground truth, the list of BMC and PLoS journals as well as the Science-Metrix journal classification.
* [das classifier folder](das_classifier): contains code and instructions to reproduce the DAS classification step.
* [dev set folder](dev_set): contains a uniform sample of 1000 articles from the PMC OA collection, created using the [sample_dev_set.py](sample_dev_set.py) script, which can be used for agile development.
* [exports folder](exports): contains exports from scripts.
* [logs folder](logs): empty, for log files.
* A set of scripts to create the dataset, see below for instructions. You might need to adjust some parameters at the beginning of each script before using them.

## Instructions

1. Download the Pubmed OA collection, e.g. via their FTP service: https://www.ncbi.nlm.nih.gov/pmc/tools/ftp. Optionally sample it using the [sample_dev_set.py](sample_dev_set.py) script (or use the development dataset of 1000 articles which is provided in the [dev set folder](dev_set)).
2. Setup a MongoDB and update the [config file](config/config.conf).
3. Run the [parser_main.py](parser_main.py) script, which will create a first collection of articles in Mongo.
4. Run the [calculate_stats.py](calculate_stats.py) script, which will calculate citation counts for articles and authors and create the relative collections in Mongo.
5. Run the [get_export.py](get_export.py) script, which will create a first export of the dataset in the [exports folder](exports).
6. Run the [get_das_unique.py](get_das_unique.py) script, which will pull out unique DAS for classification.
7. Follow the instructions in the [DAS classifier README](das_classifier/README.md).
8. Run the [get_export_merged.py](get_export_merged.py) script, to create the final dataset for analysis.
9. Optionally, run the [evaluation_plos.py](evaluation_plos.py) and [get_authors_top.py](get_authors_top.py) for evaluation.

## Requirements

Recommended Python Version: 3.9.18 
See [requirements](../requirements.txt).
