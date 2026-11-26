"""
Services of the QA report: the computation of the metrics of its sections
from products read by :mod:`..repository`, and the export of the QA
products derived during a run. Files are read and written only through
:mod:`..repository`.

``metadata_summary``
    Computes the observation parameters of a field from the timestamps
    of its measurement sets.
``flagging_statistics``
    Computes the flagging statistics of a field from its final gaintable.
``pre_flagging``
    Counts the flags of the visibilities of a measurement set before
    calibration and exports them, and computes the pre-flagging of a
    field from them.
``bandpass_fit``
    Computes the phase and gain amplitude fit quality of each station of
    a field from its bandpass gaintables.
``cable_delays``
    Computes the cable delays of a field from its delay tables.
"""
