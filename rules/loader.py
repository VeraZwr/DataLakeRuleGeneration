from rules.dictionary_rule import load_dictionary_rules
from rules.custom_rules import CUSTOM_RULES

def load_all_rules():
    return load_dictionary_rules() + CUSTOM_RULES

#load yaml format rule
import yaml

def load_rules_from_yaml(yaml_path="/Users/veraz/PycharmProjects/DataLakeRuleGeneration/rules.yaml"):
    with open(yaml_path, 'r') as f:
        rules = yaml.safe_load(f)
    return rules

