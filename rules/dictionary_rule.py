from rules.base_rule import BaseRule

# Rule profiles: include conditions + features used
SIMPLE_RULE_PROFILES = {
    "is_id": {
        "conditions": {"unique_ratio": 1.0, "null_ratio": 0.0, "semantic_domain": "rank", "basic_data_type": "integer"},
        "features": ["unique_ratio", "null_ratio", "basic_data_type", "semantic_domain"],
        "description": "All values are unique and non-null",
        "sample_column": "index"
    },
    "is_single_value": {
        "conditions": {"distinct_num": 1},
        "features": ["distinct_num"],
        "description": "Only one distinct value",
        "sample_column": "HospitalType"
    },
    #"is_primary_key": {
    #    "conditions": {"unique_ratio": 1.0, "null_ratio": 0.0},
    #    "features": ["unique_ratio", "null_ratio"],
    #    "description": "Column is a primary key (unique & non-null)"
    #},
    "is_unique": {
        "conditions": {"unique_ratio": 1.0},
        "features": ["unique_ratio"],
        "description": "All values are unique",
        "sample_column": "index"
    },
    "is_nullable": {
        "conditions": {"null_ratio": lambda x: x > 0},
        "features": ["null_ratio"],
        "description": "Contains null values",
        "sample_column": "ibu" # hospital only have empty value
    },
    "has_low_cardinality": {
        "conditions": {"unique_ratio": lambda x: x < 0.1},
        "features": ["unique_ratio"],
        "description": "Low cardinality (distinct values < 10%)",
        "sample_column": "Condition"
    },
    "value_in_range": {
        "conditions": {
            "numeric_min_value": lambda x: x >= 0,
            "numeric_max_value": lambda x: x <= 10
        },
        "features": ["numeric_min_value", "numeric_max_value"],
        "description": "Values within expected range",
        "sample_column": "RatingValue" # movies_1
    },
    "quartile_thresholds": {
        "conditions": {
            "Q1": lambda x: x >= 0,
            "Q3": lambda x: x <= 1000
        },
        "features": ["Q1", "Q3"],
        "description": "Quartile thresholds within acceptable range",
        "sample_column": ""
    },
    "length_within": {
        "conditions": {
            "min_len": lambda x: x >= 5,
            "max_len": lambda x: x <= 5
        },
        "features": ["min_len", "max_len"],
        "description": "String length within expected range"
    },
    "decimal_precision": {
        "conditions": {
            "max_decimal_num": lambda x: x <= 3
        },
        "features": ["max_decimal_num"],
        "description": "Decimal places within acceptable precision",
        "sample_column": "abv" #beers
    },
    "matches_regex": {
        "conditions": {"basic_data_type": "time_am_pm", "semantic_domain":"duration", "dominant_pattern": "^\\d:\\d\\d\\s[A-Za-z]\\.[A-Za-z]\\.$"},
        "features": ["basic_data_type", "semantic_domain", "dominant_pattern"],
        "description": "Matches expected regex pattern",
        "sample_column": "sched_dep_time"
    },
    "matches_regex": {
        "conditions": {"basic_data_type": "string", "semantic_domain": "state",
                       "dominant_pattern": "^[A-Za-z][A-Za-z]$"},
        "features": ["basic_data_type", "semantic_domain", "dominant_pattern"],
        "description": "Matches expected regex pattern",
        "sample_column": "state"
    },
    "matches_regex": {
        "conditions": {"basic_data_type": "integer", "semantic_domain": "region",
                       "dominant_pattern": "^\\d\\d\\d\\d\\d$"},
        "features": ["basic_data_type", "semantic_domain", "dominant_pattern"],
        "description": "Matches expected regex pattern",
        "sample_column": "zip"
    },
    "matches_regex": {
        "conditions": {"basic_data_type": "time_am_pm", "semantic_domain": "duration",
                       "dominant_pattern": "^\\d:\\d\\d\\s[A-Za-z]\\.[A-Za-z]\\.$"},
        "features": ["basic_data_type", "semantic_domain", "dominant_pattern"],
        "description": "Matches expected regex pattern",
        "sample_column": "sched_dep_time"
    },
    #"benford_conformity": {
    #    "conditions": {"first_digit_distribution": "benford_distribution"},
    #    "features": ["first_digit_distribution"],
    #    "description": "First digit distribution follows Benford’s law",
    #    "sample_column": ""
    #},
    #"semantic_class_is": {
    #    "conditions": {"semantic_domain": "expected_class"},
    #    "features": ["semantic_domain"],
    #    "description": "Semantic class matches expected class"
    #},
    "top_key_words_are": {
        "conditions": {"basic_data_type": "boolean", "semantic_domain":"status", "top_keywords": {"yes", "no"}},
        "features": ["basic_data_type", "semantic_domain"],
        "description": "Semantic class matches expected class",
        "sample_column": "emergency_services"
    },
    "is_english_text": {
    "conditions": {
        "words_alphabet_mean": lambda x: x > 0.5,
        "words_unique_mean": lambda x: x > 0.2,
        "null_ratio": lambda x: x < 0.5
    },
    "features": ["words_alphabet_mean", "words_unique_mean", "null_ratio"],
    "description": "Column contains mostly English alphabetic words with reasonable uniqueness",
    "sample_column": ""
}
}

class DictionaryRule(BaseRule):
    def __init__(self, name, conditions, description, features):
        self.name = name
        self.description = description
        self.conditions = conditions
        self.used_features = features
        self.expected_value = None  # for cell-level checks

    def applies(self, col):
        for k, v in self.conditions.items():
            if col.get(k) != v:
                return False
        return True

    def prepare(self, column_data):
        """
        Optional pre-check setup for cell-level validation.
        Only meaningful for 'is_single_value' and similar.
        """
        if self.name == "is_single_value":
            unique_values = list(set(column_data))
            if len(unique_values) == 1:
                self.expected_value = unique_values[0]
            else:
                self.expected_value = None

    def validate_cell(self, value):
        """
        Return True if cell is valid under this rule.
        For now, only supports 'is_single_value' explicitly.
        """
        if self.name == "is_single_value" and self.expected_value is not None:
            return value == self.expected_value
        return True  # assume valid by default

def load_dictionary_rules():
    return [
        DictionaryRule(name, profile["conditions"], profile["description"], profile["features"])
        for name, profile in SIMPLE_RULE_PROFILES.items()
    ]
