class BaseRule(object):
    name = "base_rule"
    description = "Base rule class"
    used_features = []

    def applies(self, col_profile: dict) -> bool:
        raise NotImplementedError("Subclasses must implement applies()")