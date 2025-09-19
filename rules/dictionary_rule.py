from rules.base_rule import BaseRule
import pandas as pd
import re
from rules.evaluation import has_spelling_errors


# Rule profiles: include conditions + features used
SIMPLE_RULE_PROFILES = {
    #statistic rule
    "is_id": {
        "conditions": {"unique_ratio": 1.0, "null_ratio": 0.0, "semantic_domain": "rank", "basic_data_type": "integer"},
        "features": ["unique_ratio", "null_ratio", "basic_data_type", "semantic_domain"],
        "description": "All values are unique and non-null",
        "sample_column": ["305b_Assessed_Lake_2018::objectid(long)"]
    },
    "is_single_value": {
        "conditions": {"distinct_num": 1.0},
        "features": ["distinct_num"],
        "description": "Only one distinct value",
        "sample_column": [""]#"305b_Assessed_Lake_2018::cyclevalue(long)", "305b_Assessed_Lake_2018::sizeunit", "305b_Assessed_Lake_2018::watertype","305b_Assessed_Lake_2018::drinking_water_attainment_code"]
    },

    "is_unique": {
        "conditions": {"unique_ratio": 1.0},
        "features": ["unique_ratio"],
        "description": "All values are unique",
        "sample_column": [""]#"305b_Assessed_Lake_2018::objectid(long)"]
    },
    "is_nullable": {
        "conditions": {"null_ratio": lambda x: x != 0},
        "features": ["null_ratio"],
        "description": "Cannot contains null values",
        "sample_column": ["305b_Assessed_Lake_2018::drinking_water_attainment_code"]
    },
    "is_not_nullable": {
        "conditions": {"null_ratio": 0},
        "features": ["null_ratio"],
        "description": "Contains null values",
        "sample_column": ["flights::sched_dep_time","flights::act_dep_time", "305b_Assessed_Lake_2018::aquatic_life_attainment_code", "305b_Assessed_Lake_2018::objectid(long)"]
    },
    "has_low_cardinality": {
        "conditions": {"unique_ratio": lambda x: x < 0.1},
        "features": ["unique_ratio"],
        "description": "Low cardinality (distinct values < 10%)",
        "sample_column": [""]
    },
    #dynamic rule

    "matches_regex_time": {
        "description": "Column must match a regex pattern based on its domain",
        "features": ["dominant_pattern"],
        "sample_column": ["flights::sched_dep_time", "flights::act_arr_time"],
        "conditions": {
                    # "basic_data_type": "time_am_pm",
                    # "semantic_domain": "duration",
                    "dominant_pattern": r"^\d{1,2}:\d{2}\s[A-Za-z]\.[A-Za-z]\.$"
                }
            },
    "matches_regex_ounces":{
                "sample_column": ["beers::ounces"],
                "features": ["dominant_pattern"],
                "conditions": {
                    "dominant_pattern": r"^\d{1,2}$"
                }
            },
    "matches_regex_ibu": {
                "sample_column": ["beers::ibu"],
                "features": ["dominant_pattern"],
                "conditions": {
                    "dominant_pattern": r"^\d{1,3}$"
                }
            },
    "matches_regex_provider_number": {
        "sample_column": ["hospital::provider_number"],
        "features": ["dominant_pattern"],
        "conditions": {
            "dominant_pattern": r"^\d{5}$"
        }
    },
    "matches_regex_phone": {
            "sample_column": ["hospital::phone"],
            "features": ["dominant_pattern"],
            "conditions": {
                "dominant_pattern": r"^\d{9}$"
            },
            "description": "Phone number must be an 11-digit number."
        },
    "matches_regex_assessment_unit_id": {
                "sample_column": ["305b_Assessed_Lake_2020::assessmentunitid"],
                "features": ["dominant_pattern"],
                "conditions": {
                    "dominant_pattern": r"^CT\d{4}-00-\d-L\d{1,2}_01$"
                }
            },




        #        {
        #            "sample_column": ["hospital_state", "beers_state"],
        #            "conditions": {
        #                "basic_data_type": "string",
        #                "semantic_domain": "state",
        #                "dominant_pattern": r"^[A-Za-z]{2}$"
        #            }
        #        },
        #        {
        #            "sample_column": "hospital_zip",
        #            "conditions": {
        #                "basic_data_type": "integer",
        #                "semantic_domain": "region",
        #                "dominant_pattern": r"^\d{5}$"
        #            }
        #        },
        #        {
        #            "sample_column": "305b_Assessed_Lake_2018_id_three_zero_five_b",
        #            "conditions": {
        #                "basic_data_type": "string",
        #                "semantic_domain": "code",
        #                "dominant_pattern": r"^[A-Za-z][A-Za-z]\\d\\d\\d\\d\\-\\d\\d\\-\\d\\-[A-Za-z]\\d_\\d\\d$"
        #            }
        #        }
#semantic domain check
    "is_city": {
            "conditions": {"semantic_domain": "city"},
            "features": ["semantic_domain"],
            "description": "Is a valid city name",
            "sample_column": ["beers::city", "hospital::city"]
        },
    "is_state_id": {
                "conditions": {"semantic_domain": "state"},
                "features": ["semantic_domain"],
                "description": "Is a valid state name",
                "sample_column": ["beers::state", "hospital::state"]
                },
    "is_zip": {
                    "conditions": {"semantic_domain": "region"},
                    "features": ["semantic_domain"],
                    "description": "Is a valid zip code",
                    "sample_column": ["hospital::zip"]
                },
    # Merge with patterns
    # "data_type_is": {
    #    "description": "Column must be a certain data type",
    #    "features": ["basic_data_type"],
    #    "entries": [
    #        {
    #            "sample_column": ["beers_ounces"],
    #            "conditions": {
    #                "basic_data_type": "integer"
    #            }
    #        },
    #    ]
    #},
    "value_in_range": {
        "conditions": {
            "numeric_min_value": lambda x: x >= 0,
            "numeric_max_value": lambda x: x <= 10
        },
        "features": ["numeric_min_value", "numeric_max_value"],
        "description": "Values within expected range",
        "sample_column": ["movies_1::rating_value"] # movies_1
    },
    "quartile_thresholds": {
        "conditions": {
            "Q1": lambda x: x >= 0,
            "Q3": lambda x: x <= 1000
        },
        "features": ["Q1", "Q3"],
        "description": "Quartile thresholds within acceptable range",
        "sample_column": [""]
    },
    "length_within": {
        "conditions": {
            "min_len": 1,
            "max_len": 1
        },
        "features": ["min_len", "max_len"],
        "description": "String length within expected range",
        "sample_column": ["305b_Assessed_Lake_2018::aquatic_life_attainment_code"]
    },
    "decimal_precision": {
        "conditions": {
            "max_decimal": 3
        },
        "features": ["max_decimal"],
        "description": "Decimal places within acceptable precision",
        "sample_column": ["beers::abv"] #beers
    },


    "name_format_quality_check": {
        "conditions": {
            "dominant_pattern": r"^[A-Z][a-z]+(\s[A-Z][a-z]+)*\s\([A-Z][a-z]+(?:/[A-Z][a-z]+)*\)$",
            "semantic_domain": "location"
        },
        "features": ["dominant_pattern", "semantic_domain"],
        "description": "Geographic names follow expected format with town",
        "sample_column": [""]#"305b_Assessed_Lake_2018::watername"
    },
    "is_spelled_correctly": {
        "conditions": {
            "spell_check": lambda val: not has_spelling_errors(val)
        },
        "features": ["spell_check"],
        "description": "No spelling errors in the value",
        "sample_column": ["hospital::name","hospital::address_1", "hospital::type","hospital::owner","hospital::measure_name","hospital::sample","305b_Assessed_Lake_2020::assessmentunitname","305b_Assessed_Lake_2020::locationdescription", "305b_Assessed_Lake_2020::watertypename", "305b_Assessed_Lake_2020::units","305b_Assessed_Lake_2020::useclassname","305b_Assessed_Lake_2020::ct_two_zero_two_zero_aql_use_usename","305b_Assessed_Lake_2020::ct_two_zero_two_zero_aql_use_attainment", "305b_Assessed_Lake_2020::ct_two_zero_two_zero_rec_use_usename","305b_Assessed_Lake_2020::ct_two_zero_two_zero_rec_use_attainment","305b_Assessed_Lake_2020::ct_two_zero_two_zero_fshcon_use_attainment","305b_Assessed_Lake_2020::ct_two_zero_two_zero_dw_use_usename", "305b_Assessed_Lake_2020::ct_two_zero_two_zero_dw_use_attainment","305b_Assessed_Lake_2020::impaired"]
    #"305b_Assessed_Lake_2018::locationvalue", "305b_Assessed_Lake_2018::watertype", "305b_Assessed_Lake_2018::classname","305b_Assessed_Lake_2018::fish_consumption_attainment", "305b_Assessed_Lake_2018::drinking_water_attainment",
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
    # "is_primary_key": {
    #    "conditions": {"unique_ratio": 1.0, "null_ratio": 0.0},
    #    "features": ["unique_ratio", "null_ratio"],
    #    "description": "Column is a primary key (unique & non-null)"
    # },
    "top_key_words_boolean": {
        "conditions": {"basic_data_type": "boolean", "semantic_domain":"status", "top_keywords": {"yes", "no", "Yes", "No"}},
        "features": ["basic_data_type", "semantic_domain"],
        "description": "Semantic class matches expected class",
        "sample_column": ["hospital::emergency_service"]#, "305b_Assessed_Lake_2018::impaired"]
    },
    "is_english_text": {
    "conditions": {
        "words_alphabet_mean": lambda x: x > 0.5,
        "words_unique_mean": lambda x: x > 0.2,
        "null_ratio": lambda x: x < 0.5
    },
    "features": ["words_alphabet_mean", "words_unique_mean", "null_ratio"],
    "description": "Column contains mostly English alphabetic words with reasonable uniqueness",
    "sample_column": [""]
}
}

class DictionaryRule(BaseRule):
    def __init__(self, name, conditions, description, features, sample_column):
        self.name = name
        self.description = description
        self.conditions = conditions
        self.features = features
        self.expected_value = None
        self.sample_column = sample_column
        self.regex = None
    '''
    def applies(self, col):
        if isinstance(self.sample_column, dict):
            col_name = col.get("column_name")
            if col_name not in self.sample_column:
                return False
            rule_conditions = self.sample_column[col_name]
        else:
            rule_conditions = self.conditions

        for k, v in rule_conditions.items():
            col_val = col.get(k)
            if callable(v):
                if not v(col_val):
                    return False
            elif col_val != v:
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
    '''
    def applies(self, col):
        col_name = col.get("column_name")

        # Special logic for matches_regex
        if self.name == "matches_regex":
            # Always allow apply(), true pattern will be checked in `prepare`
            return True

        # If this is a dictionary per column
        if isinstance(self.sample_column, dict):
            if col_name not in self.sample_column:
                return False
            rule_conditions = self.sample_column[col_name]
        else:
            rule_conditions = self.conditions

        for k, v in rule_conditions.items():
            col_val = col.get(k)
            if callable(v):
                if not v(col_val):
                    return False
            elif col_val != v:
                return False
        return True

    def validate_cell(self, value):
        if self.name == "is_single_value" and self.expected_value is not None:
            return value == self.expected_value

        elif self.name == "is_not_nullable":
            return not pd.isna(value)

        elif self.name == "matches_regex":
            print(f"[DEBUG] Validating cell '{value}' with regex '{self.regex.pattern if self.regex else None}'")
            if value is None or pd.isna(value) or not hasattr(self, "regex") or self.regex is None:
                return False  # nulls or invalid pattern
            return bool(self.regex.fullmatch(str(value)))

        elif self.name == "is_spelled_correctly":
            return not has_spelling_errors(str(value)) if pd.notna(value) else True

        return True

    # to process the rules
    def prepare(self, column_data, sample_column_profile=None):
        if self.name == "is_single_value":
            unique_values = list(set(column_data))
            self.expected_value = unique_values[0] if len(unique_values) == 1 else None


        elif self.name == "matches_regex":
            if sample_column_profile and sample_column_profile.get("dominant_pattern"):
                pattern = sample_column_profile["dominant_pattern"]
                self.regex = re.compile(pattern)
            else:
                non_null_values = [v for v in column_data if pd.notna(v)]
                if non_null_values:
                    inferred_pattern = self.regex_pattern_category(str(non_null_values[0]))
                    print(f"[DEBUG] Inferred regex pattern: {inferred_pattern}")
                    self.regex = re.compile(inferred_pattern)
                else:
                    self.regex = None

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


def load_dictionary_rules():
    rules = []
    for name, profile in SIMPLE_RULE_PROFILES.items():
        if "entries" in profile:  # Multi-entry dynamic rules
            for entry in profile["entries"]:
                rules.append(DictionaryRule(
                    name=name,
                    conditions=entry.get("conditions", {}),
                    description=profile.get("description", ""),
                    features=profile.get("features", []),
                    sample_column=entry.get("sample_column", "")
                ))
        else:  # Standard rule
            rules.append(DictionaryRule(
                name=name,
                conditions=profile.get("conditions", {}),
                description=profile.get("description", ""),
                features=profile.get("features", []),
                sample_column=profile.get("sample_column", "")
            ))
    return rules

