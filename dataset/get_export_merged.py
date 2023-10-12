#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Merges tabular data with classification data on DAS
__author__ = """Giovanni Colavizza"""

import codecs, csv
from collections import defaultdict

tabular_data_file = "exports/export.csv"
classification_file = "das_classifier/das_full_classified.tsv" # pick the outcome file you prefer here
tabular_data_file_final = "exports/export_full.csv"
separator = ";"
text_delim = '"'

# Manually create info on journal name cleanup
equivalence_classes = {
    "algorithms for molecular biology : amb": "algorithms for molecular biology",
    "behavioral and brain functions : bbf": "behavioral and brain functions",
    "behavioral and brain functions: bbf": "behavioral and brain functions",
    "bmc bioinformatics [electronic resource]": "bmc bioinformatics",
    "bmc cancer [electronic resource]": "bmc cancer",
    "bmc ear, nose, and throat disorders": "bmc ear, nose and throat disorders",
    "breast cancer research : bcr": "breast cancer research",
    "cardio-oncology (london, england)": "cardio-oncology",
    "cell communication and signaling : ccs": "cell communication and signaling",
    "clinical and molecular allergy : cma": "clinical and molecular allergy",
    "cost effectiveness and resource allocation : c/e": "cost effectiveness and resource allocation : c/e",
    "diabetology and metabolic syndrome": "diabetology & metabolic syndrome",
    "dynamic medicine : dm": "dynamic medicine",
    "epidemiologic perspectives & innovations : ep+i": "epidemiologic perspectives & innovations",
    "fibrogenesis and tissue repair": "fibrogenesis & tissue repair",
    "genetics, selection, evolution. : gse": "genetics, selection, evolution",
    "genetics, selection, evolution : gse": "genetics, selection, evolution",
    "immunity & ageing : i & a": "immunity & ageing",
    "implementation science : is": "implementation science",
    "international seminars in surgical oncology : isso": "international seminars in surgical oncology",
    "journal of hematology and oncology": "journal of hematology & oncology",
    "journal of trauma management and outcomes": "journal of trauma management & outcomes",
    "philosophy, ethics, and humanities in medicine : pehm": "philosophy, ethics, and humanities in medicine",
    "reproductive biology and endocrinology : rb&e": "reproductive biology and endocrinology",
    "world journal of emergency surgery : wjes": "world journal of emergency surgery"
}

def clean_journal_name(x):
    if x in equivalence_classes.keys():
        x = equivalence_classes[x]
    return x

if __name__ == "__main__":

    all_das = defaultdict(str)

    # LOAD classification
    das_counter = 0
    with codecs.open(classification_file, encoding="utf8") as f:
        for row in f.readlines():
            das, das_class = row.split("\t")
            all_das[das] = das_class
            das_counter += 1
    print(das_counter)

    # LOAD data
    no_das_class = 0
    data = list()
    with codecs.open(tabular_data_file, encoding="utf8") as f:
        reader = csv.DictReader(f, delimiter=";", quotechar='"')
        for row in reader:
            row["das_class"] = 0
            if row["das"] in all_das.keys():
                row["das_class"] = int(all_das[row["das"]].strip()) # add das_class column
            else:
                no_das_class += 1
            row["j_lower"] = clean_journal_name(row["journal"].lower()) # add clean journal name column
            data.append(row)
    print(no_das_class)

    # EXPORT main table
    fieldnames = ["pmid", "pmcid", "doi", "title", "n_authors",
                  "journal", "j_lower", "journal_domain", "journal_field",
                  "journal_subfield", "publisher_id", "is_plos", "is_bmc",
                  "n_references_tot", "n_references", "has_das",
                  "das_encouraged", "das_required", "das", "das_class",
                  "p_year", "p_month", "has_month", "n_cit_2", "n_cit_3",
                  "n_cit_5", "n_cit_tot", "h_index_min", "h_index_max",
                  "h_index_mean", "h_index_median"
                  ]
    with codecs.open(tabular_data_file_final, "w", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

    print("All done!")
