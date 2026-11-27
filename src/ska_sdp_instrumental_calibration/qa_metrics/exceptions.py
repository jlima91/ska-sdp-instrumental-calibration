"""
Exceptions of the QA metrics package.
"""

__all__ = ["RunMetadataError", "QAReportException"]


class RunMetadataError(Exception):
    """
    Run metadata of an INST run cannot be written or loaded: missing,
    malformed or incomplete
    """


class QAReportException(Exception):
    """
    A section of the QA metric report cannot be built. Shown in the report
    in place of the section, by ``sections/error_template.html``.

    Parameters
    ----------
    message : str
        What could not be built.
    cause : BaseException, optional
        Error it was caused by, shown as "Caused by" in the report.
    """

    def __init__(self, message: str, cause: BaseException | None = None):
        super().__init__(message)
        # Same chaining as "raise ... from cause", for an error that is
        # shown in the report instead of being raised
        self.__cause__ = cause


class QAValidationError(QAReportException):
    """
    Raised when a section cannot be built due to an exit status.
    """
