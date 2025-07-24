import argparse
import os
import sys
import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from dataset import Dataset
from column_features.column_name_features import ColumnNameFeature, COLUMN_CATEGORY_PROTOTYPES
from column_features.data_type_features import  DataTypeFeatures
import argparse
from doduo.doduo.doduo import Doduo
import nltk, re, string, operator
from nltk.tokenize import word_tokenize
from collections import Counter
import pandas as pd

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

    def guess_column_type(self, column_data):
        col = column_data.dropna()
        if col.empty:
            return "empty"

        col_str = col.astype(str)

        datetime_match = col_str.str.match(r"\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?$").mean()
        if datetime_match > 0.7:
            return "datetime64"

        am_pm_match = col_str.str.contains(
            r"^\s*\d{1,2}:\d{2}(?::\d{2})?\s*(?:a\.?\s*m\.?|p\.?\s*m\.?)\.?\s*$",
            flags=re.IGNORECASE
        ).mean()
        if am_pm_match > 0.6:
            return "time_am_pm"

        time_24h_match = col_str.str.match(r"^\s*\d{1,2}:\d{2}(?::\d{2})?\s*$").mean()
        if time_24h_match > 0.6:
            return "time_24h"

        date_match = col_str.str.match(r"\d{4}-\d{2}-\d{2}$").mean()
        if date_match > 0.7:
            return "date"

        percentage_match = col_str.str.endswith('%').mean()
        if percentage_match > 0.7:
            return "percentage"

        bool_vals = {"true", "false", "yes", "no", "1", "0"}
        bool_match = col_str.str.lower().isin(bool_vals).mean()
        if bool_match > 0.7:
            return "boolean"

        numeric_col = pd.to_numeric(col_str, errors='coerce')
        numeric_ratio = numeric_col.notna().mean()
        if numeric_ratio > 0.7:
            is_integer = (numeric_col.dropna() == numeric_col.dropna().astype(int)).mean()
            if is_integer > 0.7:
                return "integer"
            else:
                return "float"

        unique_count = col_str.nunique()
        total_count = len(col_str)
        if total_count > 0 and (unique_count / total_count < 0.05) and unique_count <= 20:
            return "categorical"

        inferred = pd.api.types.infer_dtype(col_str, skipna=True)
        if inferred in ["string", "unicode", "mixed"]:
            return inferred

        return "unknown"

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

    def regex_pattern_category(self, value):
        value = value.strip()
        pattern = ""
        for char in value:
            if char.isdigit():
                pattern += r"\d"
            elif char.isalpha():
                pattern += r"[A-Za-z]"
            elif char.isspace():
                pattern += r"\s"
            else:
                pattern += re.escape(char)
        return "^" + pattern + "$"

    # -------------------------
    #combined with dataset profiler
    def column_profiler(self, column_data, column_name):
        """
        Profile a single column with dataset-level stats applied per column.
        """


        col = column_data.fillna("").astype(str)
        row_num = len(col)

        stop_words_set = set(nltk.corpus.stopwords.words("english"))
        keywords_dictionary = {}

        # Basic nulls & uniqueness
        null_ratio = (column_data == "").sum() / row_num
        distinct_num = col.nunique()
        unique_ratio = distinct_num / row_num

        # Character-level stats
        char_unique = set()
        char_alphabet = char_numeric = char_punct = char_misc = 0

        # Word-level stats
        word_unique = set()
        word_alphabet = word_numeric = word_punct = word_misc = word_total_len = 0

        # Cell-level stats
        cell_unique = set()
        cell_alphabet = cell_numeric = cell_punct = cell_misc = cell_total_len = 0
        cell_null = 0

        for cell in col:
            # Characters
            for char in cell:
                char_unique.add(char)
                if re.fullmatch("[a-zA-Z]", char):
                    char_alphabet += 1
                elif re.fullmatch("[0-9]", char):
                    char_numeric += 1
                elif re.fullmatch(f"[{re.escape(string.punctuation)}]", char):
                    char_punct += 1
                else:
                    char_misc += 1

            # Words
            words = word_tokenize(cell)
            for word in words:
                word_unique.add(word)
                word_total_len += len(word)
                if re.fullmatch("[a-zA-Z_-]+", word):
                    word_alphabet += 1
                    word_lower = word.lower()
                    if word_lower not in stop_words_set:
                        keywords_dictionary[word_lower] = keywords_dictionary.get(word_lower, 0) + 1
                elif re.fullmatch("[0-9]+([.,][0-9]+)?", word):
                    word_numeric += 1
                elif re.fullmatch(f"[{re.escape(string.punctuation)}]+", word):
                    word_punct += 1
                else:
                    word_misc += 1

            # Cells
            cell_unique.add(cell)
            if re.fullmatch("[a-zA-Z _-]+", cell):
                cell_alphabet += 1
            elif re.fullmatch("[0-9]+([.,][0-9]+)?", cell):
                cell_numeric += 1
            elif re.fullmatch(f"[{re.escape(string.punctuation)}]+", cell):
                cell_punct += 1
            else:
                cell_misc += 1

            cell_total_len += len(cell)
            if cell.strip() == "":
                cell_null += 1

        # Word length mean
        word_len_avg = word_total_len / max(len(word_unique), 1)

        # Numeric stats
        try:
            col_numeric = pd.to_numeric(col, errors="coerce").dropna()
            max_digits = None
            max_decimals = None

            if not col_numeric.empty:
                digits_counts = []
                decimals_counts = []
                for num in col_numeric:
                    num_str = str(num)
                    if '.' in num_str:
                        integer_part, decimal_part = num_str.split('.')
                        digits_counts.append(
                            len(integer_part.replace('-', '').lstrip('0')) + len(decimal_part.rstrip('0')))
                        decimals_counts.append(len(decimal_part.rstrip('0')))
                    else:
                        digits_counts.append(len(num_str.replace('-', '').lstrip('0')))
                        decimals_counts.append(0)
                max_digits = max(digits_counts) if digits_counts else None
                max_decimals = max(decimals_counts) if decimals_counts else None

            numeric_min = col_numeric.min()
            numeric_max = col_numeric.max()
            most_freq_value_ratio = col_numeric.value_counts(normalize=True).iloc[0] if not col_numeric.empty else None
            q1 = col_numeric.quantile(0.25)
            q2 = col_numeric.quantile(0.50)
            q3 = col_numeric.quantile(0.75)
            first_digits = col_numeric.astype(str).str.replace(r"\D", "", regex=True).str[0]
            first_digit = Counter(first_digits).most_common(1)[0][0] if not first_digits.empty else None
        except:
            numeric_min = numeric_max = most_freq_value_ratio = q1 = q2 = q3 = first_digit = None

        # Histogram & bins
        value_counts = col.value_counts()
        histogram = value_counts.index[0] if not value_counts.empty else None
        histogram_freq = value_counts.iloc[0] if not value_counts.empty else None

        equi_width_bins = pd.cut(col_numeric, bins=10).value_counts().to_dict() if not col_numeric.empty else {}
        equi_depth_bins = pd.qcut(col_numeric, q=10,
                                  duplicates="drop").value_counts().to_dict() if not col_numeric.empty else {}
        max_equi_width_bin = max(equi_width_bins, key=equi_width_bins.get) if equi_width_bins else None
        max_equi_depth_bin = max(equi_depth_bins, key=equi_depth_bins.get) if equi_depth_bins else None

        # String length
        col_lengths = col.apply(len)
        min_len = col_lengths.min()
        max_len = col_lengths.max()
        avg_len = col_lengths.mean()

        # Pattern
        pattern_hist = Counter(col.apply(self.regex_pattern_category))
        dominant_pattern = pattern_hist.most_common(1)[0][0] if pattern_hist else None


        return {
            "column_name": self.file_name +"_"+column_name,
            "row_num": row_num,
            "null_ratio": null_ratio,
            "distinct_num": distinct_num,
            "unique_ratio": unique_ratio,

            "characters_unique": len(char_unique),
            "characters_alphabet": char_alphabet,
            "characters_numeric": char_numeric,
            "characters_punctuation": char_punct,
            "characters_miscellaneous": char_misc,

            "words_unique": len(word_unique),
            "words_alphabet": word_alphabet,
            "words_numeric": word_numeric,
            "words_punctuation": word_punct,
            "words_miscellaneous": word_misc,
            "words_length_avg": word_len_avg,

            "cells_unique": len(cell_unique),
            "cells_alphabet": cell_alphabet,
            "cells_numeric": cell_numeric,
            "cells_punctuation": cell_punct,
            "cells_miscellaneous": cell_misc,
            "cells_length_avg": cell_total_len / row_num,
            "cells_null": cell_null,

            "top_keywords": dict(
                sorted(keywords_dictionary.items(), key=lambda x: x[1], reverse=True)[:self.KEYWORDS_COUNT_PER_COLUMN]),

            "numeric_min": numeric_min,
            "numeric_max": numeric_max,
            "max_digits": max_digits,
            "max_decimals": max_decimals,
            "Q1": q1,
            "Q2": q2,
            "Q3": q3,
            "most_freq_value_ratio": most_freq_value_ratio,
            "first_digit": first_digit,

            "histogram": histogram,
            "histogram_freq": histogram_freq,
            "equi_width_bin": max_equi_width_bin,
            "equi_depth_bin": max_equi_depth_bin,

            "max_len": max_len,
            "min_len": min_len,
            "avg_len": avg_len,

            "dominant_pattern": dominant_pattern,
            "basic_data_type": self.guess_column_type(column_data)

            #"basic_data_type": pd.api.types.infer_dtype(column_data, skipna=True),
        }

    """
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
    """
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
            return np.mean(np.array(columns_value_list).astype(float) / d.dataframe.shape[0])

        def g(columns_value_list):
            return np.var(np.array(columns_value_list).astype(float) / d.dataframe.shape[0])

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
    def train_column_regression(self, all_column_profiles, performance_labels):
        """
        Train regression models to predict strategy performance per column.
        :param all_column_profiles: list of column profiles (dict)
        :param performance_labels: list of target performance metrics (dict per strategy)
        """
        # Example: predict F1 score per strategy
        feature_names = [k for k in all_column_profiles[0].keys() if isinstance(all_column_profiles[0][k], (int, float))]

        X = np.array([[profile[feat] for feat in feature_names] for profile in all_column_profiles])
        regressors = {}

        for strategy_name in performance_labels[0].keys():
            y = np.array([perf[strategy_name] for perf in performance_labels])
            model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
            model.fit(X, y)
            regressors[strategy_name] = model
            print(f"Trained regressor for {strategy_name}")

        self.column_regressors = regressors
        self.column_features = feature_names

    def predict_strategy_performance(self, column_profile):
        """
        Predict strategy performance for a new column profile.
        """
        X = np.array([[column_profile[feat] for feat in self.column_features]])
        predictions = {strategy: model.predict(X)[0] for strategy, model in self.column_regressors.items()}
        return predictions

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