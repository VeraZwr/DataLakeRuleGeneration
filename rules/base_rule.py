import pandas as pd
from typing import List
class BaseRule(object):
    name = "base_rule"
    description = "Base rule class"
    used_features = []

    def applies(self, col_profile: dict) -> bool:
        raise NotImplementedError("Subclasses must implement applies()")

    def validate_cell(self, value):
        return True