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
