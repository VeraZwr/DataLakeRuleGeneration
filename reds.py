import os
import sys
import re
import time
import math
import string
import random
import operator
import pickle
import numpy
import nltk
import sklearn.neighbors
import sklearn.naive_bayes
import sklearn.tree
import sklearn.svm
import sklearn.neural_network
import sklearn.kernel_ridge
import sklearn.ensemble
from dataset import Dataset
from column_features.column_name_features import ColumnNameFeature, COLUMN_CATEGORY_PROTOTYPES
from column_features.data_type_features import  DataTypeFeatures
########################################
class REDS:
    """
    The main class.
    """

    def __init__(self, datasets_folder="datasets/Quintet", results_folder="results"):
        self.DATASETS_FOLDER = datasets_folder
        self.RESULTS_FOLDER = results_folder
        self.DATASETS = {}
        self.KEYWORDS_COUNT_PER_COLUMN = 10
        self.colname_transformer = ColumnNameFeature(category_prototypes=COLUMN_CATEGORY_PROTOTYPES)
        self.colname_transformer.fit()
        self.dtype_transformer = DataTypeFeatures()

    def load_datasets(self):
        self.DATASETS["beers"] = {
            "name": "beers",
            "path": os.path.join(self.DATASETS_FOLDER, "beers", "dirty.csv"),
            "clean_path": os.path.join(self.DATASETS_FOLDER, "beers", "clean.csv"),
            "functions": [...],
            "patterns": [...],
        }

    def dataset_profiler(self, dataset_dictionary):
        """
        This method profiles the dataset.
        """
        print(dataset_dictionary)
        d = Dataset(dataset_dictionary)
        column_name_cat_list = [0.0] * d.dataframe.shape[1]
        data_type_list = [0.0] * d.dataframe.shape[1]


        print ("Profiling dataset {}...".format(d.dataframe))
        current_column_names = d.dataframe.columns.tolist()
        column_name_cat_list = self.colname_transformer.transform(current_column_names)
        data_type_list = self.dtype_transformer.transform(d.dataframe)



        characters_unique_list = [0.0] * d.dataframe.shape[1]
        characters_alphabet_list = [0.0] * d.dataframe.shape[1]
        characters_numeric_list = [0.0] * d.dataframe.shape[1]
        characters_punctuation_list = [0.0] * d.dataframe.shape[1]
        characters_miscellaneous_list = [0.0] * d.dataframe.shape[1]
        words_unique_list = [0.0] * d.dataframe.shape[1]
        words_alphabet_list = [0.0] * d.dataframe.shape[1]
        words_numeric_list = [0.0] * d.dataframe.shape[1]
        words_punctuation_list = [0.0] * d.dataframe.shape[1]
        words_miscellaneous_list = [0.0] * d.dataframe.shape[1]
        words_length_list = [0.0] * d.dataframe.shape[1]
        cells_unique_list = [0.0] * d.dataframe.shape[1]
        cells_alphabet_list = [0.0] * d.dataframe.shape[1]
        cells_numeric_list = [0.0] * d.dataframe.shape[1]
        cells_punctuation_list = [0.0] * d.dataframe.shape[1]
        cells_miscellaneous_list = [0.0] * d.dataframe.shape[1]
        cells_length_list = [0.0] * d.dataframe.shape[1]
        cells_null_list = [0.0] * d.dataframe.shape[1]
        top_keywords_dictionary = {a.lower(): 1.0 for a in d.dataframe.columns}
        print(top_keywords_dictionary)
        stop_words_set = set(nltk.corpus.stopwords.words("english"))


        for column, attribute in enumerate(d.dataframe.columns):
            characters_dictionary = {}
            words_dictionary = {}
            cells_dictionary = {}
            keywords_dictionary = {}
            for cell in d.dataframe[attribute]:
                for character in cell:
                    if character not in characters_dictionary:
                        characters_dictionary[character] = 0
                        characters_unique_list[column] += 1
                    characters_dictionary[character] += 1
                    if re.findall("^[a-zA-Z]$", character):
                        characters_alphabet_list[column] += 1
                    elif re.findall("^[0-9]$", character):
                        characters_numeric_list[column] += 1
                    elif re.findall("^[{}]$".format(string.punctuation), character):
                        characters_punctuation_list[column] += 1
                    else:
                        characters_miscellaneous_list[column] += 1
                for word in nltk.word_tokenize(cell):
                    if word not in words_dictionary:
                        words_dictionary[word] = 0
                        words_unique_list[column] += 1
                    words_dictionary[word] += 1
                    if re.findall("^[a-zA-Z_-]+$", word):
                        words_alphabet_list[column] += 1
                        word = word.lower()
                        if word not in keywords_dictionary:
                            keywords_dictionary[word] = 0
                        keywords_dictionary[word] += 1
                    elif re.findall("^[0-9]+[.,][0-9]+$", word) or re.findall("^[0-9]+$", word):
                        words_numeric_list[column] += 1
                    elif re.findall("^[{}]+$".format(string.punctuation), word):
                        words_punctuation_list[column] += 1
                    else:
                        words_miscellaneous_list[column] += 1
                    words_length_list[column] += len(word)
                if cell not in cells_dictionary:
                    cells_dictionary[cell] = 0
                    cells_unique_list[column] += 1
                cells_dictionary[cell] += 1
                if re.findall("^[a-zA-Z_ -]+$", cell):
                    cells_alphabet_list[column] += 1
                elif re.findall("^[0-9]+[.,][0-9]+$", cell) or re.findall("^[0-9]+$", cell):
                    cells_numeric_list[column] += 1
                elif re.findall("^[{}]+$".format(string.punctuation), cell, re.IGNORECASE):
                    cells_punctuation_list[column] += 1
                else:
                    cells_miscellaneous_list[column] += 1
                cells_length_list[column] += len(cell)
                if cell == "":
                    cells_null_list[column] += 1
            if sum(words_dictionary.values()) > 0:
                words_length_list[column] /= sum(words_dictionary.values())
            sorted_keywords_dictionary = sorted(keywords_dictionary.items(), key=operator.itemgetter(1), reverse=True)
            for keyword, frequency in sorted_keywords_dictionary[:self.KEYWORDS_COUNT_PER_COLUMN]:
                if keyword not in stop_words_set:
                    top_keywords_dictionary[keyword] = float(frequency) / d.dataframe.shape[0]

        def f(columns_value_list):
            return numpy.mean(numpy.array(columns_value_list).astype(float) / d.dataframe.shape[0])

        def g(columns_value_list):
            return numpy.var(numpy.array(columns_value_list).astype(float) / d.dataframe.shape[0])

        dataset_profile = {
            #"dataset_column_names_cat": column_name_cat_list,
            #"dominant_data_type": data_type_list,
            "dataset_top_keywords": top_keywords_dictionary,
            "dataset_rules_count": len(self.DATASETS[d.name]["functions"]),
            "dataset_patterns_count": len(self.DATASETS[d.name]["patterns"]),
            "characters_unique_mean": f(characters_unique_list),
            "characters_unique_variance": g(characters_unique_list),
            "characters_alphabet_mean": f(characters_alphabet_list),
            "characters_alphabet_variance": g(characters_alphabet_list),
            "characters_numeric_mean": f(characters_numeric_list),
            "characters_numeric_variance": g(characters_numeric_list),
            "characters_punctuation_mean": f(characters_punctuation_list),
            "characters_punctuation_variance": g(characters_punctuation_list),
            "characters_miscellaneous_mean": f(characters_miscellaneous_list),
            "characters_miscellaneous_variance": g(characters_miscellaneous_list),
            "words_unique_mean": f(words_unique_list),
            "words_unique_variance": g(words_unique_list),
            "words_alphabet_mean": f(words_alphabet_list),
            "words_alphabet_variance": g(words_alphabet_list),
            "words_numeric_mean": f(words_numeric_list),
            "words_numeric_variance": g(words_numeric_list),
            "words_punctuation_mean": f(words_punctuation_list),
            "words_punctuation_variance": g(words_punctuation_list),
            "words_miscellaneous_mean": f(words_miscellaneous_list),
            "words_miscellaneous_variance": g(words_miscellaneous_list),
            "words_length_mean": f(words_length_list),
            "words_length_variance": g(words_length_list),
            "cells_unique_mean": f(cells_unique_list),
            "cells_unique_variance": g(cells_unique_list),
            "cells_alphabet_mean": f(cells_alphabet_list),
            "cells_alphabet_variance": g(cells_alphabet_list),
            "cells_numeric_mean": f(cells_numeric_list),
            "cells_numeric_variance": g(cells_numeric_list),
            "cells_punctuation_mean": f(cells_punctuation_list),
            "cells_punctuation_variance": g(cells_punctuation_list),
            "cells_miscellaneous_mean": f(cells_miscellaneous_list),
            "cells_miscellaneous_variance": g(cells_miscellaneous_list),
            "cells_length_mean": f(cells_length_list),
            "cells_length_variance": g(cells_length_list),
            "cells_null_mean": f(cells_null_list),
            "cells_null_variance": g(cells_null_list)
        }
        print (dataset_profile)
        pickle.dump(dataset_profile, open(os.path.join(self.RESULTS_FOLDER, d.name, "dataset_profile.dictionary"), "wb"))

########################################
if __name__ == "__main__":
    # ----------------------------------------
    application = REDS()
    application.load_datasets()
    # ----------------------------------------
    for dd in application.DATASETS.values():
        print ("===================== Dataset: {} =====================".format(dd["name"]))
        if not os.path.exists(os.path.join(application.RESULTS_FOLDER, dd["name"])):
            os.mkdir(os.path.join(application.RESULTS_FOLDER, dd["name"]))
        application.dataset_profiler(dd)
        # application.strategy_profiler(dd)
        # application.evaluation_profiler(dd)