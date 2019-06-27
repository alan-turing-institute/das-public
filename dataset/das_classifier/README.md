# Classifier of Data Availability Statements

This folder contains code and data to classify DAS. Proceed as follows:

1. Make sure you reach the point when you have run [get_export](../get_export.py) and have created the first [export file](../exports/export.csv).
2. Run the [get_das_unique.py](../get_das_unique.py) script to create the [das_full.csv](input/das_full.csv) dataset containing unique DAS to classify.
3. Run the [classify_das.py](classify_das.py) script here to train and compare classifiers.
4. Pick a classifier, name it as `das_classifier/output/das_full_classified.tsv` and run the [get_export_merged](../get_export_merged.py) script to export the final dataset for analysis.

## Contents

* [classify_das.py](classify_das.py) contains the script to train a variety of classifiers, output their results as well as an evaluation table to allow for a comparison across.
* [das_full_classified.tsv](das_full_classified.tsv) contains the classification result used in the paper.
* INPUT:
    - [das_full.csv](input/das_full.csv) list of das and their frequency in the corpus, created with the script [get_das_unique.py](../get_das_unique.py).
    - [das_journal.csv](input/das_journal.csv) list of das and their frequency in the corpus, grouped by journal, created with the script [get_das_unique.py](../get_das_unique.py).
    - [das_full_annotation.csv](input/das_full_annotation.csv) manually annotated dataset used for training/testing.
    - [glove embeddings (50d)](input/glove.6B.50d.txt.zip) glove embeddings used for one of the classifier (50d, compressed, please unzip before use).
* OUTPUT: this folder will contain classification of DAS contained in [das_full.csv](input/das_full.csv) for all models, as well as the [overview file](output/overview_models_parameters.csv) to compare them (we provide an example of the latter).

## Requirements

See [requirements](../requirements.txt).