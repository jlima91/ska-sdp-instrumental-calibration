class Tagger:
    def __init__(self, tag):
        self._tag = tag

    def __contains__(self, func):
        return getattr(func, "__prism_tag__", None) == self._tag

    def __call__(self, func):
        func.__prism_tag__ = self._tag
        return func


class Tags:
    BROADCASTER = Tagger("vectorize:broadcast")
    AGGREGATOR = Tagger("vectorize:aggregate")
