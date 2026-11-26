"""
Repository of the QA report: the run manifest, in which the stages record
the products of a run, and the writers, lookups and readers of those
products.

``manifest_schema``, ``run_manifest``, ``field_metadata``
    Schema of the run manifest, the manifest of a run and its records for
    one field.
``metadata``
    Looks up the stations and measurement sets of a field.
``gaintable``
    Looks up the final and bandpass gaintables of a field and reads them.
``delaytable``
    Looks up the delay tables of a field and reads them.
``pre_flagging``
    Writes the pre-flagging data file of a measurement set, and looks up
    and reads those of a field.
``gain_fits``
    Writes the curve fits produced by gain flagging to a Zarr store.

A lookup returns an empty list where a product with one path per
measurement set is not recorded: its reader raises a ``RunMetadataError``
for it. A missing single product, such as the final gaintable, is raised
by its lookup.
"""
