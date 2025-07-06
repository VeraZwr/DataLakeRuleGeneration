import argparse
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
import pandas as pd
from collections import Counter
from dataset import Dataset
from column_features.column_name_features import ColumnNameFeature, COLUMN_CATEGORY_PROTOTYPES
from column_features.data_type_features import  DataTypeFeatures
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from doduo.doduo.doduo import Doduo

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DODUO_DIR = os.path.join(BASE_DIR, "..", "doduo")
sys.path.append(DODUO_DIR)

########################################
class REDS:
    """
    The main class.
    """

    def __init__(self, datasets_folder="datasets/Quintet", results_folder="results"):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.datasets_folder = datasets_folder
        self.DATASETS_FOLDER = os.path.join(base_dir, datasets_folder)
        self.RESULTS_FOLDER = os.path.join(base_dir, results_folder)
        self.DATASETS = {}
        self.KEYWORDS_COUNT_PER_COLUMN = 10
        self.colname_transformer = ColumnNameFeature(category_prototypes=COLUMN_CATEGORY_PROTOTYPES)
        self.colname_transformer.fit()
        self.dtype_transformer = DataTypeFeatures()
        self.file_name = "rayyan"
        args = argparse.Namespace()
        args.model = "viznet"
        self.doduo = Doduo(args)

    def load_datasets(self):
        self.DATASETS[self.file_name] = {
            "name": self.file_name,
            "path": os.path.join(self.DATASETS_FOLDER, self.file_name, "dirty.csv"),
            "clean_path": os.path.join(self.DATASETS_FOLDER, self.file_name, "clean.csv"),
            "functions": [...],
            "patterns": [...],
        }

    def guess_column_types(file_path, delimiter=',', has_headers=True):
        try:
            # Read the CSV file using the specified delimiter and header settings
            df = pd.read_csv(file_path, sep=delimiter, header=0 if has_headers else None)

            # Initialize a dictionary to store column data types
            column_types = {}

            # Loop through columns and infer data types
            for column in df.columns:
                # sample_values = df[column].dropna().sample(min(5, len(df[column])), random_state=42)

                # Check for datetime format "YYYY-MM-DD HH:MM:SS"
                is_datetime = all(re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', str(value)) for value in df[column])

                # Check for date format "YYYY-MM-DD"
                is_date = all(re.match(r'\d{4}-\d{2}-\d{2}', str(value)) for value in df[column])

                # Assign data type based on format detection
                if is_datetime:
                    inferred_type = 'datetime64'
                elif is_date:
                    inferred_type = 'date'
                else:
                    inferred_type = pd.api.types.infer_dtype(df[column], skipna=True)

                column_types[column] = inferred_type

            return (True, column_types)  # Return success and column types
        except pd.errors.ParserError:
            return (False, str(e))  # Return error message

    def guess_semantic_domain_doduo(self, df):
        adf = self.doduo.annotate_columns(df)
        return adf.coltypes, adf.valid_col_indices

    ###########################################################
    @staticmethod
    def generalize_pattern(value):
        import string
        if pd.isnull(value):
            return "NULL"
        pattern = []
        for char in str(value):
            if char.isdigit():
                pattern.append("0")
            elif char.isalpha():
                pattern.append("A")
            elif char in string.punctuation:
                pattern.append(char)  # <-- special chars kept AS-IS
            elif char.isspace():
                pattern.append("_")
            else:
                pattern.append("?")
        return "".join(pattern)

    # -------------------------
    def column_profiler(self, column_data, column_name):
        col = column_data.dropna() #remove nan or none
        row_num = len(column_data)
        null_ratio = column_data.isnull().sum() / float(row_num)
        distinct_num = col.nunique()
        unique_ratio = distinct_num / float(row_num)
        max_digit_num = None
        max_decimal_num = None
        most_frequent_first_digit = None
        # Type Inference.cannot get dominant type in messy column, only say it is mixed
        #TODO a method to get dominant type
        inferred_type = pd.api.types.infer_dtype(col, skipna=True)
        try:
            col_numeric = pd.to_numeric(col)
            numeric_min = col_numeric.min()
            numeric_max = col_numeric.max()
            most_freq_value_ratio = col_numeric.value_counts(normalize=True).iloc[0]
            q1 = col_numeric.quantile(0.25)
            q2 = col_numeric.quantile(0.50)
            q3 = col_numeric.quantile(0.75)
            # First digit distribution (Benford)
            first_digits = col_numeric.astype(str).str.replace(r"\D", "", regex=True).str[0]
            first_digit_dist = Counter(first_digits)
            most_frequent_first_digit = first_digit_dist.most_common(1)[0][0]

            col_numeric_to_string = col_numeric.dropna().astype(str)
            max_digit_num = col_numeric_to_string.apply(lambda x: len(x.split(".")[0].replace("-",""))).max()
            max_decimal_num = col_numeric_to_string.apply(lambda x: len(x.split(".")[1] if "." in x else 0)).max()
        except:
            numeric_min = numeric_max = q1 = q2 = q3 = most_freq_value_ratio = None
            first_digit_dist = {}

        # Histogram (value counts, as fallback)
        value_counts = col.value_counts()
        histogram = value_counts.index[0] if not value_counts.empty else None
        histogram_freq = value_counts.iloc[0] if not value_counts.empty else None

        # Equi-width histogram (only if numeric)
        #equi_width_bins = []
        #equi_depth_bins = []
        #if pd.api.types.is_numeric_dtype(col):
        #    equi_width_bins = pd.cut(col, bins=10).value_counts().to_dict()
        #    equi_depth_bins = pd.qcut(col, q=10, duplicates="drop").value_counts().to_dict()
        # Equi-width / depth → keep largest bin only
        equi_width_bins = {}
        equi_depth_bins = {}
        max_equi_width_bin = None
        max_equi_depth_bin = None

        if pd.api.types.is_numeric_dtype(col):
            equi_width_bins = pd.cut(col, bins=10).value_counts().to_dict()
            equi_depth_bins = pd.qcut(col, q=10, duplicates="drop").value_counts().to_dict()
            if equi_width_bins:
                max_equi_width_bin = max(equi_width_bins, key=equi_width_bins.get)
            if equi_depth_bins:
                max_equi_depth_bin = max(equi_depth_bins, key=equi_depth_bins.get)
        #str len
        col_str = col.astype(str)
        value_len = col_str.apply(len)
        min_len = value_len.min()
        max_len = value_len.max()
        avg_len = value_len.mean()

        #patterns histogram
        value_patterns = col_str.apply(self.generalize_pattern)
        #pattern_histogram = dict(Counter(value_patterns))
        pattern_histogram = Counter(col_str.apply(self.generalize_pattern))
        dominant_pattern = pattern_histogram.most_common(1)[0][0] if pattern_histogram else None

        #DBMS types
        # dBoost
        #col_df = pd.DataFrame({column_name: col})
        #rel = dboost.Relation.load_from_dataframe(col_df)
        #model = dboost.DBoost()
        #model.fit(rel)
        #scores = model.detect_errors()
        #anomaly_ratio = (scores > 0.5).sum() / float(len(scores))  # 0 is normal, 1 is error

        column_profile = {
            "column_name": column_name,
            "row_num": row_num,
            "null_ratio": null_ratio,
            "distinct_num": distinct_num,
            "unique_ratio": unique_ratio,
            "histogram": histogram,
            "histogram_freq": histogram_freq,
            "equi_width_bin": max_equi_width_bin,
            "equi_depth_bin": max_equi_depth_bin,
            "numeric_min_value": numeric_min,
            "numeric_max_value": numeric_max,
            "most_freq_value_ratio": most_freq_value_ratio,
            "Q1": q1,
            "Q2": q2,
            "Q3": q3,
            "first_digit": most_frequent_first_digit,
            "basic_data_type": inferred_type,
            "max_digit_num": max_digit_num,
            "max_decimal_num": max_decimal_num,
            "max_len": max_len,
            "min_len": min_len,
            "avg_len": avg_len,
            "domaint_pattern": dominant_pattern
            #"dboost_anomaly_ratio": anomaly_ratio
        }

        #if semantic_domain_guess_doduo is not None:
         #   column_profile["semantic_domain_guess_doduo"] = semantic_domain_guess_doduo
        return column_profile
    ################################################
    def dataset_profiler(self, dataset_dictionary):
        """
        This method profiles the dataset.
        """
        # print(dataset_dictionary)
        d = Dataset(dataset_dictionary)
        #df = d.dataframe
        #semantic_guesses_doduo = self.guess_semantic_domain_doduo(df)
        #column_profiles = []
        #for i, column_name in enumerate(df.columns):
        #    column_profile = self.column_profiler(
        #        df[column_name],
        #        column_name,
        #        semantic_domain_guess_doduo=semantic_guesses_doduo[i]
        #    )
         #   column_profiles.append(column_profile)
        #semantic_guesses_doduo = self.guess_semantic_domain_doduo(df)

        column_name_cat_list = [0.0] * d.dataframe.shape[1]
        data_type_list = [0.0] * d.dataframe.shape[1]
        column_index = [0.0] * d.dataframe.shape[1]


        print ("Profiling dataset {}...".format(d.dataframe))
        current_column_names = d.dataframe.columns.tolist()
        column_index = {name: idx for idx, name in enumerate(current_column_names)}
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
        semantic_guesses_doduo, valid_col_indices = self.guess_semantic_domain_doduo(d.dataframe)

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
                for word in word_tokenize(cell):
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
            #"dataset_column_index": column_index,
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
        # print (dataset_profile)
        pickle.dump(dataset_profile, open(os.path.join(self.RESULTS_FOLDER, d.name, "dataset_profile.dictionary"), "wb"))

        # build column profile
        column_profiles = []
        valid_indices = list(range(len(semantic_guesses_doduo)))  # adjust if needed
        semantic_list =[]
        for i, column_name in enumerate(d.dataframe.columns):
            column_profile = self.column_profiler(d.dataframe[column_name], column_name)
            if i in valid_indices:
                column_profile["semantic_domain"] = semantic_guesses_doduo[valid_indices.index(i)]
                semantic_list.append(column_name+": "+ semantic_guesses_doduo[valid_indices.index(i)])
            else:
                column_profile["semantic_domain"] = None
            column_profiles.append(column_profile)
        print(column_profiles)
        #print("test",semantic_list)
        # Save column-level profile in a separate file
        pickle.dump(column_profiles, open(os.path.join(self.RESULTS_FOLDER, d.name, "column_profile.dictionary"), "wb"))


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