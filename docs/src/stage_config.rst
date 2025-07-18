Stages and configurations
#########################

.. This page is generated using docs/generate_config.py

The descriptions of each stage are copied from the docstrings of stages.
Refer to the `API page for stages <package/guide.html#stages>`_

Each stage has parameters, which are defined in the YAML config file passed to the pipeline.


load_data
*********

    Load the Measurement Set data.

Parameters
==========

..  table::
    :width: 100%
    :widths: 15, 10, 10, 45, 10, 10

    +------------------+--------+-----------+--------------------------------------------------------------------------------+------------+------------------------------------------+
    | Param            | Type   | Default   | Description                                                                    | Nullable   | Allowed values                           |
    +==================+========+===========+================================================================================+============+==========================================+
    | fchunk           | int    | 32        | Number of frequency channels per chunk                                         | True       |                                          |
    +------------------+--------+-----------+--------------------------------------------------------------------------------+------------+------------------------------------------+
    | ack              | bool   | False     | Ask casacore to acknowledge each table operation                               | True       |                                          |
    +------------------+--------+-----------+--------------------------------------------------------------------------------+------------+------------------------------------------+
    | start_chan       | int    | 0         | Starting channel to read                                                       | True       |                                          |
    +------------------+--------+-----------+--------------------------------------------------------------------------------+------------+------------------------------------------+
    | end_chan         | int    | 0         | End channel to read                                                            | True       |                                          |
    +------------------+--------+-----------+--------------------------------------------------------------------------------+------------+------------------------------------------+
    | datacolumn       | str    | DATA      | MS data column to read DATA, CORRECTED_DATA, or                     MODEL_DATA | True       | ['DATA', 'CORRECTED_DATA', 'MODEL_DATA'] |
    +------------------+--------+-----------+--------------------------------------------------------------------------------+------------+------------------------------------------+
    | selected_sources | list   | ``null``  | Sources to select                                                              | True       |                                          |
    +------------------+--------+-----------+--------------------------------------------------------------------------------+------------+------------------------------------------+
    | selected_dds     | list   | ``null``  | Data descriptors to select                                                     | True       |                                          |
    +------------------+--------+-----------+--------------------------------------------------------------------------------+------------+------------------------------------------+
    | average_channels | bool   | False     | Average all channels read                                                      | True       |                                          |
    +------------------+--------+-----------+--------------------------------------------------------------------------------+------------+------------------------------------------+


predict_vis
***********

    Predict model visibilities using a local sky model.

Parameters
==========

..  table::
    :width: 100%
    :widths: 15, 10, 10, 45, 10, 10

    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+-------------------------------------+------------------+
    | Param                    | Type   | Default   | Description                                                                      | Nullable                            | Allowed values   |
    +==========================+========+===========+==================================================================================+=====================================+==================+
    | beam_type                | str    | everybeam | Type of beam model to use. Default is 'everybeam'                                | True                                |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+-------------------------------------+------------------+
    | normalise_at_beam_centre | bool   | False     | If true, before running calibration, multiply vis             and model vis by   | True                                |                  |
    |                          |        |           | the inverse of the beam response in the             beam pointing direction.     |                                     |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+-------------------------------------+------------------+
    | eb_ms                    | str    | ``null``  | If beam_type is "everybeam" but input ms does             not have all of the    | True                                |                  |
    |                          |        |           | metadata required by everybeam, this parameter             is used to specify a  |                                     |                  |
    |                          |        |           | separate dataset to use when setting up             the beam models.             |                                     |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+-------------------------------------+------------------+
    | eb_coeffs                | str    | ``null``  | Everybeam coeffs datadir containing beam             coefficients. Required if   | True                                |                  |
    |                          |        |           | bbeam_type is 'everybeam'.                                                       |                                     |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+-------------------------------------+------------------+
    | gleamfile                | str    | ``null``  | Specifies the location of gleam catalogue             file gleamegc.dat          | True                                |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+-------------------------------------+------------------+
    | lsm_csv_path             | str    | ``null``  | Specifies the location of CSV file containing the             sky model. The CSV | True                                |                  |
    |                          |        |           | file should be in OSKAR CSV format.                                              |                                     |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+-------------------------------------+------------------+
    | fov                      | float  | 10.0      | Specifies the width of the cone used when             searching for compoents,   | True                                |                  |
    |                          |        |           | in units of degrees. Default: 10.                                                |                                     |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+-------------------------------------+------------------+
    | flux_limit               | float  | 1.0       | Specifies the flux density limit used when             searching for compoents,  | True                                |                  |
    |                          |        |           | in units of Jy. Defaults to 1                                                    |                                     |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+-------------------------------------+------------------+
    | alpha0                   | float  | -0.78     | Nominal alpha value to use when fitted data             are unspecified. Default | True                                |                  |
    |                          |        |           | is -0.78.                                                                        |                                     |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+-------------------------------------+------------------+
    | reset_vis                | bool   | False     | Whether or not to set visibilities to zero before             accumulating       | True                                |                  |
    |                          |        |           | components. Default is False.                                                    |                                     |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+-------------------------------------+------------------+
    | export_model_vis         | bool   | False     | ``null``                                                                         | Export predicted model visibilities |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+-------------------------------------+------------------+


bandpass_calibration
********************

    Performs Bandpass Calibration

Parameters
==========

..  table::
    :width: 100%
    :widths: 15, 10, 10, 45, 10, 10

    +-----------------------------------+--------+-------------------+---------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | Param                             | Type   | Default           | Description                                                                     | Nullable   | Allowed values                                                                             |
    +===================================+========+===================+=================================================================================+============+============================================================================================+
    | run_solver_config.solver          | str    | gain_substitution | Calibration algorithm to use.                 (default="gain_substitution")     | True       | ['gain_substitution', 'jones_substitution', 'normal_equations', 'normal_equations_presum'] |
    |                                   |        |                   | Options are:                 "gain_substitution" - original substitution        |            |                                                                                            |
    |                                   |        |                   | algorithm                 with separate solutions for each polarisation term.   |            |                                                                                            |
    |                                   |        |                   | "jones_substitution" - solve antenna-based Jones matrices                 as a  |            |                                                                                            |
    |                                   |        |                   | whole, with independent updates within each iteration.                          |            |                                                                                            |
    |                                   |        |                   | "normal_equations" - solve normal equations within                 each         |            |                                                                                            |
    |                                   |        |                   | iteration formed from linearisation with respect to                 antenna-    |            |                                                                                            |
    |                                   |        |                   | based gain and leakage terms.                 "normal_equations_presum" - same  |            |                                                                                            |
    |                                   |        |                   | as normal_equations                 option but with an initial accumulation of  |            |                                                                                            |
    |                                   |        |                   | visibility                 products over time and frequency for each solution   |            |                                                                                            |
    |                                   |        |                   | interval. This can be much faster for large datasets                 and        |            |                                                                                            |
    |                                   |        |                   | solution intervals.                                                             |            |                                                                                            |
    +-----------------------------------+--------+-------------------+---------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.refant          | int    | 0                 | Reference antenna.                 Currently only activated for                 | False      |                                                                                            |
    |                                   |        |                   | gain_substitution solver                                                        |            |                                                                                            |
    +-----------------------------------+--------+-------------------+---------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.niter           | int    | 50                | Number of solver iterations.                                                    | False      |                                                                                            |
    +-----------------------------------+--------+-------------------+---------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.phase_only      | bool   | False             | Solve only for the phases. This can be set                 to ``True`` when     | False      |                                                                                            |
    |                                   |        |                   | solver is "gain_substitution",                 otherwise it must be ``False``.  |            |                                                                                            |
    +-----------------------------------+--------+-------------------+---------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.tol             | float  | 1e-06             | Iteration stops when the fractional change                 in the gain solution | False      |                                                                                            |
    |                                   |        |                   | is below this tolerance.                                                        |            |                                                                                            |
    +-----------------------------------+--------+-------------------+---------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.crosspol        | bool   | False             | Do solutions including cross polarisations                 i.e. XY, YX or RL,   | False      |                                                                                            |
    |                                   |        |                   | LR.                 Only used by "gain_substitution" solver.                    |            |                                                                                            |
    +-----------------------------------+--------+-------------------+---------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.normalise_gains | str    | ``null``          | Normalises the gains.                 Only available when solver is             | True       | ['``null``', 'mean', 'median']                                                             |
    |                                   |        |                   | "gain_substitution".                 Possible types of normalization are:       |            |                                                                                            |
    |                                   |        |                   | "mean", "median".                 To perform no normalization, set this to      |            |                                                                                            |
    |                                   |        |                   | ``null``.                                                                       |            |                                                                                            |
    +-----------------------------------+--------+-------------------+---------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.jones_type      | str    | T                 | Type of calibration matrix T or G or B.                 The frequency axis of   | False      | ['T', 'G', 'B']                                                                            |
    |                                   |        |                   | the output GainTable                 depends on the value provided:             |            |                                                                                            |
    |                                   |        |                   | "B": the output frequency axis is the same as                 that of the input |            |                                                                                            |
    |                                   |        |                   | Visibility.                 "T" or "G": the solution is assumed to be           |            |                                                                                            |
    |                                   |        |                   | frequency-independent, and the frequency axis of the                 output     |            |                                                                                            |
    |                                   |        |                   | contains a single value: the average frequency                 of the input     |            |                                                                                            |
    |                                   |        |                   | Visibility's channels.                                                          |            |                                                                                            |
    +-----------------------------------+--------+-------------------+---------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.timeslice       | float  | ``null``          | Defines time scale over which each gain solution                 is valid. This | True       |                                                                                            |
    |                                   |        |                   | is used to define time axis of the GainTable.                 This parameter is |            |                                                                                            |
    |                                   |        |                   | interpreted as follows,                  float: this is a custom time interval  |            |                                                                                            |
    |                                   |        |                   | in seconds.                 Input timestamps are grouped by intervals of this   |            |                                                                                            |
    |                                   |        |                   | duration,                 and said groups are separately averaged to produce    |            |                                                                                            |
    |                                   |        |                   | the output time axis.                  ``None``: match the time resolution of   |            |                                                                                            |
    |                                   |        |                   | the input, i.e. copy                 the time axis of the input Visibility      |            |                                                                                            |
    +-----------------------------------+--------+-------------------+---------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | plot_config.plot_table            | bool   | False             | Plot the generated gaintable                                                    | False      |                                                                                            |
    +-----------------------------------+--------+-------------------+---------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | plot_config.fixed_axis            | bool   | False             | Limit amplitude axis to [0-1]                                                   | False      |                                                                                            |
    +-----------------------------------+--------+-------------------+---------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | flagging                          | bool   | False             | Run RFI flagging                                                                | False      |                                                                                            |
    +-----------------------------------+--------+-------------------+---------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | visibility_key                    | str    | vis               | Visibility data to be used for calibration.                                     | True       | ['vis', 'corrected_vis']                                                                   |
    +-----------------------------------+--------+-------------------+---------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | export_gaintable                  | bool   | False             | Export intermediate gain solutions.                                             | False      |                                                                                            |
    +-----------------------------------+--------+-------------------+---------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+


generate_channel_rm
*******************

    Generates channel rotation measures

Parameters
==========

..  table::
    :width: 100%
    :widths: 15, 10, 10, 45, 10, 10

    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | Param                             | Type   | Default           | Description                                                                      | Nullable   | Allowed values                                                                             |
    +===================================+========+===================+==================================================================================+============+============================================================================================+
    | fchunk                            | int    | -1                | Number of frequency channels per chunk.             If set to -1, use fchunk     | True       |                                                                                            |
    |                                   |        |                   | value from load_data                                                             |            |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | peak_threshold                    | float  | 0.5               | Height of peak in the RM spectrum required             for a rotation detection. | True       |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | refine_fit                        | bool   | True              | Whether or not to refine the RM spectrum             peak locations with a       | True       |                                                                                            |
    |                                   |        |                   | nonlinear optimisation of             the station RM values.                     |            |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | visibility_key                    | str    | vis               | Visibility data to be used for calibration.                                      | True       | ['vis', 'corrected_vis']                                                                   |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | plot_rm_config.plot_rm            | bool   | False             | Plot the estimated rotational measures                 per station               | True       |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | plot_rm_config.station            | int    | 0                 | Station number to be plotted                                                     | True       |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | plot_table                        | bool   | False             | Plot the generated gain table                                                    | True       |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.solver          | str    | gain_substitution | Calibration algorithm to use.                 (default="gain_substitution")      | True       | ['gain_substitution', 'jones_substitution', 'normal_equations', 'normal_equations_presum'] |
    |                                   |        |                   | Options are:                 "gain_substitution" - original substitution         |            |                                                                                            |
    |                                   |        |                   | algorithm                 with separate solutions for each polarisation term.    |            |                                                                                            |
    |                                   |        |                   | "jones_substitution" - solve antenna-based Jones matrices                 as a   |            |                                                                                            |
    |                                   |        |                   | whole, with independent updates within each iteration.                           |            |                                                                                            |
    |                                   |        |                   | "normal_equations" - solve normal equations within                 each          |            |                                                                                            |
    |                                   |        |                   | iteration formed from linearisation with respect to                 antenna-     |            |                                                                                            |
    |                                   |        |                   | based gain and leakage terms.                 "normal_equations_presum" - same   |            |                                                                                            |
    |                                   |        |                   | as normal_equations                 option but with an initial accumulation of   |            |                                                                                            |
    |                                   |        |                   | visibility                 products over time and frequency for each solution    |            |                                                                                            |
    |                                   |        |                   | interval. This can be much faster for large datasets                 and         |            |                                                                                            |
    |                                   |        |                   | solution intervals.                                                              |            |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.refant          | int    | 0                 | Reference antenna.                 Currently only activated for                  | False      |                                                                                            |
    |                                   |        |                   | gain_substitution solver                                                         |            |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.niter           | int    | 50                | Number of solver iterations.                                                     | False      |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.phase_only      | bool   | False             | Solve only for the phases. This can be set                 to ``True`` when      | False      |                                                                                            |
    |                                   |        |                   | solver is "gain_substitution",                 otherwise it must be ``False``.   |            |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.tol             | float  | 1e-06             | Iteration stops when the fractional change                 in the gain solution  | False      |                                                                                            |
    |                                   |        |                   | is below this tolerance.                                                         |            |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.crosspol        | bool   | False             | Do solutions including cross polarisations                 i.e. XY, YX or RL,    | False      |                                                                                            |
    |                                   |        |                   | LR.                 Only used by "gain_substitution" solver.                     |            |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.normalise_gains | str    | ``null``          | Normalises the gains.                 Only available when solver is              | True       | ['``null``', 'mean', 'median']                                                             |
    |                                   |        |                   | "gain_substitution".                 Possible types of normalization are:        |            |                                                                                            |
    |                                   |        |                   | "mean", "median".                 To perform no normalization, set this to       |            |                                                                                            |
    |                                   |        |                   | ``null``.                                                                        |            |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.jones_type      | str    | T                 | Type of calibration matrix T or G or B.                 The frequency axis of    | False      | ['T', 'G', 'B']                                                                            |
    |                                   |        |                   | the output GainTable                 depends on the value provided:              |            |                                                                                            |
    |                                   |        |                   | "B": the output frequency axis is the same as                 that of the input  |            |                                                                                            |
    |                                   |        |                   | Visibility.                 "T" or "G": the solution is assumed to be            |            |                                                                                            |
    |                                   |        |                   | frequency-independent, and the frequency axis of the                 output      |            |                                                                                            |
    |                                   |        |                   | contains a single value: the average frequency                 of the input      |            |                                                                                            |
    |                                   |        |                   | Visibility's channels.                                                           |            |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.timeslice       | float  | ``null``          | Defines time scale over which each gain solution                 is valid. This  | True       |                                                                                            |
    |                                   |        |                   | is used to define time axis of the GainTable.                 This parameter is  |            |                                                                                            |
    |                                   |        |                   | interpreted as follows,                  float: this is a custom time interval   |            |                                                                                            |
    |                                   |        |                   | in seconds.                 Input timestamps are grouped by intervals of this    |            |                                                                                            |
    |                                   |        |                   | duration,                 and said groups are separately averaged to produce     |            |                                                                                            |
    |                                   |        |                   | the output time axis.                  ``None``: match the time resolution of    |            |                                                                                            |
    |                                   |        |                   | the input, i.e. copy                 the time axis of the input Visibility       |            |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | export_gaintable                  | bool   | False             | Export intermediate gain solutions.                                              | False      |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+


delay_calibration
*****************

    Performs delay calibration

Parameters
==========

..  table::
    :width: 100%
    :widths: 15, 10, 10, 45, 10, 10

    +------------------------------+--------+-----------+--------------------------------------------------------------+------------+------------------+
    | Param                        | Type   | Default   | Description                                                  | Nullable   | Allowed values   |
    +==============================+========+===========+==============================================================+============+==================+
    | oversample                   | int    | 16        | Oversample rate                                              | True       |                  |
    +------------------------------+--------+-----------+--------------------------------------------------------------+------------+------------------+
    | plot_config.plot_table       | bool   | False     | Plot the generated gaintable                                 | True       |                  |
    +------------------------------+--------+-----------+--------------------------------------------------------------+------------+------------------+
    | plot_config.fixed_axis       | bool   | False     | Limit amplitude axis to [0-1]                                | True       |                  |
    +------------------------------+--------+-----------+--------------------------------------------------------------+------------+------------------+
    | plot_config.anotate_stations | bool   | False     | Show station labels in delay                 vs station plot | True       |                  |
    +------------------------------+--------+-----------+--------------------------------------------------------------+------------+------------------+
    | export_gaintable             | bool   | False     | Export intermediate gain solutions.                          | False      |                  |
    +------------------------------+--------+-----------+--------------------------------------------------------------+------------+------------------+


smooth_gain_solution
********************

    Smooth the gain solution.

Parameters
==========

..  table::
    :width: 100%
    :widths: 15, 10, 10, 45, 10, 10

    +------------------------------+--------+---------------+------------------------------------------+------------+--------------------+
    | Param                        | Type   | Default       | Description                              | Nullable   | Allowed values     |
    +==============================+========+===============+==========================================+============+====================+
    | window_size                  | int    | 1             | Sliding window size.                     | False      |                    |
    +------------------------------+--------+---------------+------------------------------------------+------------+--------------------+
    | mode                         | str    | median        | Mode of smoothing                        | False      | ['mean', 'median'] |
    +------------------------------+--------+---------------+------------------------------------------+------------+--------------------+
    | plot_config.plot_table       | bool   | False         | Plot the smoothed gaintable              | False      |                    |
    +------------------------------+--------+---------------+------------------------------------------+------------+--------------------+
    | plot_config.plot_path_prefix | str    | smoothed-gain | Path prefix to store smoothed gain plots | False      |                    |
    +------------------------------+--------+---------------+------------------------------------------+------------+--------------------+
    | plot_config.plot_title       | str    | Smoothed Gain | Title for smoothed gain plots            | False      |                    |
    +------------------------------+--------+---------------+------------------------------------------+------------+--------------------+
    | export_gaintable             | bool   | False         | Export intermediate gain solutions.      | False      |                    |
    +------------------------------+--------+---------------+------------------------------------------+------------+--------------------+


export_visibilities
*******************

    Apply gaintable and export visibilities.

Parameters
==========

..  table::
    :width: 100%
    :widths: 15, 10, 10, 45, 10, 10

    +------------------------+--------+-----------+------------------------+------------+----------------------------------------+
    | Param                  | Type   | Default   | Description            | Nullable   | Allowed values                         |
    +========================+========+===========+========================+============+========================================+
    | data_to_export         | str    | ``null``  | Visibilities to export | True       | ['all', 'vis', 'modelvis', '``null``'] |
    +------------------------+--------+-----------+------------------------+------------+----------------------------------------+
    | apply_gaintable_to_vis | bool   | False     | Apply gaintable to vis | True       |                                        |
    +------------------------+--------+-----------+------------------------+------------+----------------------------------------+


export_gain_table
*****************

    Export gain table solutions to a file.

Parameters
==========

..  table::
    :width: 100%
    :widths: 15, 10, 10, 45, 10, 10

    +-----------------+--------+-----------+----------------------------------------+------------+--------------------+
    | Param           | Type   | Default   | Description                            | Nullable   | Allowed values     |
    +=================+========+===========+========================================+============+====================+
    | file_name       | str    | gaintable | Gain table file name without extension | True       |                    |
    +-----------------+--------+-----------+----------------------------------------+------------+--------------------+
    | export_format   | str    | h5parm    | Export file format                     | True       | ['h5parm', 'hdf5'] |
    +-----------------+--------+-----------+----------------------------------------+------------+--------------------+
    | export_metadata | bool   | False     | Export metadata into YAML file         | True       |                    |
    +-----------------+--------+-----------+----------------------------------------+------------+--------------------+


