class BaseRule(object):
    def __init__(self, rule):
        self.rule = rule

    def execute(self, data):
        raise NotImplementedError