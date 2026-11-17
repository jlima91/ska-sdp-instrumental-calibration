__all__ = ["RequiredArgumentMissingException", "QAMetadataError"]


class RequiredArgumentMissingException(RuntimeError):
    pass


class QAMetadataError(Exception):
    """
    QA metadata of an INST run cannot be loaded: missing, malformed or
    incomplete
    """
