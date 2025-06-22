from rules.base_rule import BaseRule

class IsMostlyUniqueRule(BaseRule):
    name = "is_mostly_unique"
    description = "At least 95% of values are unique and <5% null"
    used_features = ["unique_ratio", "null_ratio"]

    def applies(self, col):
        return col.get("unique_ratio", 0) >= 0.95 and col.get("null_ratio", 1) < 0.05


class IsNumericRangeRule(BaseRule):
    name = "is_numeric_range"
    description = "Has numeric min/max and Q3 - Q1 > 0"
    used_features = ["numeric_min_value", "numeric_max_value", "Q1", "Q3"]

    def applies(self, col):
        q1, q3 = col.get("Q1"), col.get("Q3")
        return (
            col.get("numeric_min_value") is not None and
            col.get("numeric_max_value") is not None and
            q1 is not None and q3 is not None and
            (q3 - q1) > 0
        )

CUSTOM_RULES = [IsMostlyUniqueRule(), IsNumericRangeRule()]
