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
        self.used_features = features  # Required for clustering

    def applies(self, col):
        for k, v in self.conditions.items():
            if col.get(k) != v:
                return False
        return True

def load_dictionary_rules():
    return [
        DictionaryRule(name, profile["conditions"], profile["description"], profile["features"])
        for name, profile in SIMPLE_RULE_PROFILES.items()
    ]
