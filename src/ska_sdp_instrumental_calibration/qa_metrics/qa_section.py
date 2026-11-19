"""
QA sections of the run metadata and their data labels, e.g.
``QASection.METADATA_SUMMARY.MS``.
"""

from enum import StrEnum


class QASectionLabel(StrEnum):
    """
    Base for the data labels of a QA section in the run metadata.

    The section key is derived from the subclass name, e.g.
    ``METADATA_SUMMARY`` -> ``metadata_summary``.
    """

    @property
    def section(self) -> str:
        """
        Key of the QA section in the run metadata.

        Returns
        -------
        str
            Lower-cased name of the QA section class.
        """
        return type(self).__name__.lower()


class QASection:
    """
    Namespace of the QA sections in the run metadata, e.g.
    ``QASection.METADATA_SUMMARY.MS``.
    """

    class METADATA_SUMMARY(QASectionLabel):
        """Data labels of the ``metadata_summary`` QA section."""

        MS = "MS"
        TIMESTAMPS = "timestamps"
        FINAL_GAINTABLE = "final_gaintable"
        CHANNELS = "channels"

    class FLAGGING(QASectionLabel):
        """Data labels of the ``flagging`` QA section."""

        PRE_FLAGGING = "pre_flagging"
        GAIN_FITS = "gain_fits"
        GAINTABLE = "gaintable"

    class BANDPASS(QASectionLabel):
        """Data labels of the ``bandpass`` QA section."""

        GAINTABLE = "gaintable"

    class DELAY(QASectionLabel):
        """Data labels of the ``delay`` QA section."""

        GAINTABLE = "gaintable"
        DELAYTABLE = "delaytable"
