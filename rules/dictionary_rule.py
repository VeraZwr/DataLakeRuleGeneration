from rules.base_rule import BaseRule

# Rule profiles: include conditions + features used
SIMPLE_RULE_PROFILES = {
    "is_identical": {
        "conditions": {"unique_ratio": 1.0, "null_ratio": 0.0},
        "features": ["unique_ratio", "null_ratio"],
        "description": "All values are unique and non-null"
    },
    "is_single_value": {
        "conditions": {"distinct_num": 1},
        "features": ["distinct_num"],
        "description": "Only one distinct value"
    },
    "is_primary_key": {
        "conditions": {"unique_ratio": 1.0, "null_ratio": 0.0},
        "features": ["unique_ratio", "null_ratio"],
        "description": "Column is a primary key (unique & non-null)"
    },
    "is_unique": {
        "conditions": {"unique_ratio": 1.0},
        "features": ["unique_ratio"],
        "description": "All values are unique"
    },
    "is_nullable": {
        "conditions": {"null_ratio": lambda x: x > 0},
        "features": ["null_ratio"],
        "description": "Contains null values"
    },
    "has_low_cardinality": {
        "conditions": {"unique_ratio": lambda x: x < 0.1},
        "features": ["unique_ratio"],
        "description": "Low cardinality (distinct values < 10%)"
    },
    "value_in_range": {
        "conditions": {
            "numeric_min_value": lambda x: x >= 0,  # placeholder, adjust domain
            "numeric_max_value": lambda x: x <= 1000  # placeholder
        },
        "features": ["numeric_min_value", "numeric_max_value"],
        "description": "Values within expected range"
    },
    "quartile_thresholds": {
        "conditions": {
            "Q1": lambda x: x >= 0,  # placeholder thresholds
            "Q3": lambda x: x <= 1000
        },
        "features": ["Q1", "Q3"],
        "description": "Quartile thresholds within acceptable range"
    },
    "length_within": {
        "conditions": {
            "min_len": lambda x: x >= 3,  # placeholder
            "max_len": lambda x: x <= 50
        },
        "features": ["min_len", "max_len"],
        "description": "String length within expected range"
    },
    "decimal_precision": {
        "conditions": {
            "max_decimal_num": lambda x: x <= 3
        },
        "features": ["max_decimal_num"],
        "description": "Decimal places within acceptable precision"
    },
    "matches_regex": {
        "conditions": {"pattern_histogram": "expected_pattern"},  # placeholder
        "features": ["pattern_histogram"],
        "description": "Matches expected regex pattern"
    },
    "benford_conformity": {
        "conditions": {"first_digit_distribution": "benford_distribution"},
        "features": ["first_digit_distribution"],
        "description": "First digit distribution follows Benford’s law"
    },
    "semantic_class_is": {
        "conditions": {"semantic_type": "expected_class"},
        "features": ["semantic_type"],
        "description": "Semantic class matches expected class"
    },
"is_english_text": {
    "conditions": {
        "words_alphabet_mean": lambda x: x > 0.5,
        "words_unique_mean": lambda x: x > 0.2,
        "null_ratio": lambda x: x < 0.5
    },
    "features": ["words_alphabet_mean", "words_unique_mean", "null_ratio"],
    "description": "Column contains mostly English alphabetic words with reasonable uniqueness"
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
