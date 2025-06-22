from rules.dictionary_rule import load_dictionary_rules
from rules.custom_rules import CUSTOM_RULES

def load_all_rules():
    return load_dictionary_rules() + CUSTOM_RULES
