Stages and configurations
#########################

.. This page is generated using docs/generate_config.py

The descriptions of each stage are copied from the docstrings of stages.
Refer to the `API page for stages <package/guide.html#stages>`_

Each stage has parameters, which are defined in the YAML config file passed to the pipeline.

Instrumental Calibration Stages
*******************************

This section describes the stages used in the Instrumental Calibration pipeline.

load_data
=========

    This stage loads the visibility data from either (in order of preference):

    1. An existing dataset stored as a zarr file inside the 'cache_directory'.
    2. From input MSv2 measurement set. Here it will create an intemediate
       zarr file with chunks along frequency and use it as input to the
       pipeline. This zarr dataset will be stored in 'cache_directory' for
       later use.

Parameters
----------

..  table::
    :width: 100%
    :widths: 15, 10, 10, 45, 10, 10

    +---------------------+--------+-----------+---------------------------------------------------------------------------------+------------+------------------------------------------+
    | Param               | Type   | Default   | Description                                                                     | Nullable   | Allowed values                           |
    +=====================+========+===========+=================================================================================+============+==========================================+
    | nchannels_per_chunk | int    | 32        | Number of frequency channels per chunk in the             written zarr file.    | False      |                                          |
    |                     |        |           | This is also the size of frequency chunk             used across the pipeline.  |            |                                          |
    +---------------------+--------+-----------+---------------------------------------------------------------------------------+------------+------------------------------------------+
    | ntimes_per_ms_chunk | int    | 5         | Number of time slots to include in each chunk             while reading from    | False      |                                          |
    |                     |        |           | measurement set.                                                                |            |                                          |
    +---------------------+--------+-----------+---------------------------------------------------------------------------------+------------+------------------------------------------+
    | cache_directory     | str    | ``null``  | Cache directory containing previously stored             visibility datasets as | True       |                                          |
    |                     |        |           | zarr files. The directory should contain             a subdirectory with same   |            |                                          |
    |                     |        |           | name as the input ms file name, which             internally contains the zarr  |            |                                          |
    |                     |        |           | and pickle files.             If None, the input ms will be converted to zarr   |            |                                          |
    |                     |        |           | file,             and this zarr file will be stored in a new 'cache'            |            |                                          |
    |                     |        |           | subdirectory under the provided output directory.                               |            |                                          |
    +---------------------+--------+-----------+---------------------------------------------------------------------------------+------------+------------------------------------------+
    | ack                 | bool   | False     | Ask casacore to acknowledge each table operation                                | False      |                                          |
    +---------------------+--------+-----------+---------------------------------------------------------------------------------+------------+------------------------------------------+
    | datacolumn          | str    | DATA      | MS data column to read visibility data from.                                    | False      | ['DATA', 'CORRECTED_DATA', 'MODEL_DATA'] |
    +---------------------+--------+-----------+---------------------------------------------------------------------------------+------------+------------------------------------------+
    | field_id            | int    | 0         | Field ID of the data in measurement set                                         | False      |                                          |
    +---------------------+--------+-----------+---------------------------------------------------------------------------------+------------+------------------------------------------+
    | data_desc_id        | int    | 0         | Data Description ID of the data in measurement set                              | False      |                                          |
    +---------------------+--------+-----------+---------------------------------------------------------------------------------+------------+------------------------------------------+


predict_vis
===========

    Predict model visibilities using a local sky model.

Parameters
----------

..  table::
    :width: 100%
    :widths: 15, 10, 10, 45, 10, 10

    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | Param                    | Type   | Default   | Description                                                                      | Nullable   | Allowed values   |
    +==========================+========+===========+==================================================================================+============+==================+
    | beam_type                | str    | everybeam | Type of beam model to use.                                                       | True       |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | normalise_at_beam_centre | bool   | True      | If true, before running calibration, multiply vis             and model vis by   | True       |                  |
    |                          |        |           | the inverse of the beam response in the             beam pointing direction.     |            |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | eb_ms                    | str    | ``null``  | If beam_type is "everybeam" but input ms does             not have all of the    | True       |                  |
    |                          |        |           | metadata required by everybeam, this parameter             is used to specify a  |            |                  |
    |                          |        |           | separate dataset to use when setting up             the beam models.             |            |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | eb_coeffs                | str    | ``null``  | Everybeam coeffs datadir containing beam             coefficients. Required if   | True       |                  |
    |                          |        |           | beam_type is 'everybeam'.                                                        |            |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | gleamfile                | str    | ``null``  | Specifies the location of gleam catalogue             file gleamegc.dat          | True       |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | lsm_csv_path             | str    | ``null``  | Specifies the location of CSV file containing the             sky model. The CSV | True       |                  |
    |                          |        |           | file should be in OSKAR CSV format.                                              |            |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | fov                      | float  | 10.0      | Specifies the width of the cone used when             searching for compoents,   | True       |                  |
    |                          |        |           | in units of degrees.                                                             |            |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | flux_limit               | float  | 1.0       | Specifies the flux density limit used when             searching for compoents,  | True       |                  |
    |                          |        |           | in units of Jy.                                                                  |            |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | alpha0                   | float  | -0.78     | Nominal alpha value to use when fitted data             are unspecified..        | True       |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+


ionospheric_delay
=================

    Calculates and applies ionospheric delay corrections to visibility data.

    This function uses an IonosphericSolver to model phase screens based on
    the difference between observed visibilities and model visibilities. It
    derives a gain table representing these phase corrections and applies it
    to the visibility data. The resulting gain table can be optionally
    exported to an H5parm file.

Parameters
----------

..  table::
    :width: 100%
    :widths: 15, 10, 10, 45, 10, 10

    +------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | Param            | Type   | Default   | Description                                                                      | Nullable   | Allowed values   |
    +==================+========+===========+==================================================================================+============+==================+
    | cluster_indexes  | list   | ``null``  | Array of integers assigning each antenna to a cluster. If None, all antennas are | True       |                  |
    |                  |        |           | treated as a single cluster                                                      |            |                  |
    +------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | block_diagonal   | bool   | True      | If True, solve for all clusters simultaneously assuming a block-diagonal system. | False      |                  |
    |                  |        |           | If False, solve for each cluster sequentially                                    |            |                  |
    +------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | niter            | int    | 500       | Number of solver iterations.                                                     | False      |                  |
    +------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | tol              | float  | 1e-06     | Iteration stops when the fractional change             in the gain solution is   | False      |                  |
    |                  |        |           | below this tolerance.                                                            |            |                  |
    +------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | zernike_limit    | int    | ``null``  | The maximum order of Zernike polynomials to use for the screen model.            | True       |                  |
    +------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | plot_table       | bool   | False     | Plot all station Phase vs Frequency                                              | False      |                  |
    +------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | export_gaintable | bool   | False     | Export intermediate gain solutions.                                              | False      |                  |
    +------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+


bandpass_calibration
====================

    Performs Bandpass Calibration

Parameters
----------

..  table::
    :width: 100%
    :widths: 15, 10, 10, 45, 10, 10

    +-----------------------------------+----------------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | Param                             | Type           | Default           | Description                                                                      | Nullable   | Allowed values                                                                             |
    +===================================+================+===================+==================================================================================+============+============================================================================================+
    | run_solver_config.solver          | str            | gain_substitution | Calibration algorithm to use. Options are:                 "gain_substitution" - | True       | ['gain_substitution', 'jones_substitution', 'normal_equations', 'normal_equations_presum'] |
    |                                   |                |                   | original substitution algorithm                 with separate solutions for each |            |                                                                                            |
    |                                   |                |                   | polarisation term.                 "jones_substitution" - solve antenna-based    |            |                                                                                            |
    |                                   |                |                   | Jones matrices                 as a whole, with independent updates within each  |            |                                                                                            |
    |                                   |                |                   | iteration.                 "normal_equations" - solve normal equations within    |            |                                                                                            |
    |                                   |                |                   | each iteration formed from linearisation with respect to                         |            |                                                                                            |
    |                                   |                |                   | antenna-based gain and leakage terms.                 "normal_equations_presum"  |            |                                                                                            |
    |                                   |                |                   | - same as normal_equations                 option but with an initial            |            |                                                                                            |
    |                                   |                |                   | accumulation of visibility                 products over time and frequency for  |            |                                                                                            |
    |                                   |                |                   | each solution                 interval. This can be much faster for large        |            |                                                                                            |
    |                                   |                |                   | datasets                 and solution intervals.                                 |            |                                                                                            |
    +-----------------------------------+----------------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.refant          | ['int', 'str'] | 0                 | Reference antenna.                 Currently only activated for                  | False      |                                                                                            |
    |                                   |                |                   | gain_substitution solver                                                         |            |                                                                                            |
    +-----------------------------------+----------------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.niter           | int            | 200               | Number of solver iterations.                                                     | False      |                                                                                            |
    +-----------------------------------+----------------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.phase_only      | bool           | False             | Solve only for the phases. This can be set                 to ``True`` when      | False      |                                                                                            |
    |                                   |                |                   | solver is "gain_substitution",                 otherwise it must be ``False``.   |            |                                                                                            |
    +-----------------------------------+----------------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.tol             | float          | 1e-06             | Iteration stops when the fractional change                 in the gain solution  | False      |                                                                                            |
    |                                   |                |                   | is below this tolerance.                                                         |            |                                                                                            |
    +-----------------------------------+----------------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.crosspol        | bool           | False             | Do solutions including cross polarisations                 i.e. XY, YX or RL,    | False      |                                                                                            |
    |                                   |                |                   | LR.                 Only used by "gain_substitution" solver.                     |            |                                                                                            |
    +-----------------------------------+----------------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.normalise_gains | str            | ``null``          | Normalises the gains.                 Only available when solver is              | True       | ['``null``', 'mean', 'median']                                                             |
    |                                   |                |                   | "gain_substitution".                 Possible types of normalization are:        |            |                                                                                            |
    |                                   |                |                   | "mean", "median".                 To perform no normalization, set this to       |            |                                                                                            |
    |                                   |                |                   | ``null``.                                                                        |            |                                                                                            |
    +-----------------------------------+----------------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.timeslice       | float          | ``null``          | Defines time scale over which each gain solution                 is valid. This  | True       |                                                                                            |
    |                                   |                |                   | is used to define time axis of the GainTable.                 This parameter is  |            |                                                                                            |
    |                                   |                |                   | interpreted as follows,                  float: this is a custom time interval   |            |                                                                                            |
    |                                   |                |                   | in seconds.                 Input timestamps are grouped by intervals of this    |            |                                                                                            |
    |                                   |                |                   | duration,                 and said groups are separately averaged to produce     |            |                                                                                            |
    |                                   |                |                   | the output time axis.                  ``None``: match the time resolution of    |            |                                                                                            |
    |                                   |                |                   | the input, i.e. copy                 the time axis of the input Visibility       |            |                                                                                            |
    +-----------------------------------+----------------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | plot_config.plot_table            | bool           | False             | Plot the generated gaintable                                                     | False      |                                                                                            |
    +-----------------------------------+----------------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | plot_config.fixed_axis            | bool           | False             | Limit amplitude axis to [0-1]                                                    | False      |                                                                                            |
    +-----------------------------------+----------------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | visibility_key                    | str            | vis               | Visibility data to be used for calibration.                                      | True       | ['vis', 'corrected_vis']                                                                   |
    +-----------------------------------+----------------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | export_gaintable                  | bool           | False             | Export intermediate gain solutions.                                              | False      |                                                                                            |
    +-----------------------------------+----------------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+


flag_gain
=========

    Performs flagging on gains and updates the weight.

Parameters
----------

..  table::
    :width: 100%
    :widths: 15, 10, 10, 45, 10, 10

    +----------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+--------------------------------+
    | Param                      | Type   | Default   | Description                                                                      | Nullable   | Allowed values                 |
    +============================+========+===========+==================================================================================+============+================================+
    | soltype                    | str    | both      | Solution type. There is a potential edge case where cyclic phases my get flagged | True       | ['phase', 'amplitude', 'both'] |
    |                            |        |           | as outliers. eg -180 and 180                                                     |            |                                |
    +----------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+--------------------------------+
    | mode                       | str    | smooth    | Detrending/fitting algorithm: smooth / poly                                      | True       | ['smooth', 'poly']             |
    +----------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+--------------------------------+
    | order                      | int    | 3         | Order of the function fitted during detrending.                                  | True       |                                |
    +----------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+--------------------------------+
    | apply_flag                 | bool   | True      | Weights are applied to the gains                                                 | True       |                                |
    +----------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+--------------------------------+
    | skip_cross_pol             | bool   | True      | Cross polarizations is skipped when flagging                                     | True       |                                |
    +----------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+--------------------------------+
    | max_ncycles                | int    | 5         | Max number of independent flagging cycles                                        | True       |                                |
    +----------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+--------------------------------+
    | n_sigma                    | float  | 10.0      | Flag values greated than n_simga * sigma_hat.             Where sigma_hat is     | True       |                                |
    |                            |        |           | 1.4826 * MeanAbsoluteDeviation.                                                  |            |                                |
    +----------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+--------------------------------+
    | n_sigma_rolling            | float  | 10.0      | Do a running rms and then flag those regions             that have a rms higher  | True       |                                |
    |                            |        |           | than n_sigma_rolling*MAD(rmses).                                                 |            |                                |
    +----------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+--------------------------------+
    | window_size                | int    | 11        | Window size for running rms                                                      | True       |                                |
    +----------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+--------------------------------+
    | normalize_gains            | bool   | True      | Normailize the amplitude and phase before flagging.                              | True       |                                |
    +----------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+--------------------------------+
    | export_gaintable           | bool   | False     | Export intermediate gain solutions.                                              | False      |                                |
    +----------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+--------------------------------+
    | plot_config.curve_fit_plot | bool   | True      | Plot the fitted curve of gain flagging                                           | False      |                                |
    +----------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+--------------------------------+
    | plot_config.gain_flag_plot | bool   | True      | Plot the flagged weights                                                         | False      |                                |
    +----------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+--------------------------------+


generate_channel_rm
===================

    Generates channel rotation measures

Parameters
----------

..  table::
    :width: 100%
    :widths: 15, 10, 10, 45, 10, 10

    +-----------------------------------+----------------+--------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | Param                             | Type           | Default            | Description                                                                      | Nullable   | Allowed values                                                                             |
    +===================================+================+====================+==================================================================================+============+============================================================================================+
    | oversample                        | int            | 5                  | Oversampling value used in the rotation             calculatiosn. Note that      | True       |                                                                                            |
    |                                   |                |                    | setting this value to some higher             integer may result in high memory  |            |                                                                                            |
    |                                   |                |                    | usage.                                                                           |            |                                                                                            |
    +-----------------------------------+----------------+--------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | peak_threshold                    | float          | 0.5                | Height of peak in the RM spectrum required             for a rotation detection. | True       |                                                                                            |
    +-----------------------------------+----------------+--------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | refine_fit                        | bool           | True               | Whether or not to refine the RM spectrum             peak locations with a       | True       |                                                                                            |
    |                                   |                |                    | nonlinear optimisation of             the station RM values.                     |            |                                                                                            |
    +-----------------------------------+----------------+--------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | visibility_key                    | str            | vis                | Visibility data to be used for calibration.                                      | True       | ['vis', 'corrected_vis']                                                                   |
    +-----------------------------------+----------------+--------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | plot_rm_config.plot_rm            | bool           | False              | Plot the estimated rotational measures                 per station               | True       |                                                                                            |
    +-----------------------------------+----------------+--------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | plot_rm_config.station            | ['int', 'str'] | 0                  | Station number/name to be plotted                                                | True       |                                                                                            |
    +-----------------------------------+----------------+--------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | plot_table                        | bool           | False              | Plot the generated gain table                                                    | True       |                                                                                            |
    +-----------------------------------+----------------+--------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.solver          | str            | jones_substitution | Calibration algorithm to use.                 Options are:                       | True       | ['gain_substitution', 'jones_substitution', 'normal_equations', 'normal_equations_presum'] |
    |                                   |                |                    | "gain_substitution" - original substitution algorithm                 with       |            |                                                                                            |
    |                                   |                |                    | separate solutions for each polarisation term.                                   |            |                                                                                            |
    |                                   |                |                    | "jones_substitution" - solve antenna-based Jones matrices                 as a   |            |                                                                                            |
    |                                   |                |                    | whole, with independent updates within each iteration.                           |            |                                                                                            |
    |                                   |                |                    | "normal_equations" - solve normal equations within                 each          |            |                                                                                            |
    |                                   |                |                    | iteration formed from linearisation with respect to                 antenna-     |            |                                                                                            |
    |                                   |                |                    | based gain and leakage terms.                 "normal_equations_presum" - same   |            |                                                                                            |
    |                                   |                |                    | as normal_equations                 option but with an initial accumulation of   |            |                                                                                            |
    |                                   |                |                    | visibility                 products over time and frequency for each solution    |            |                                                                                            |
    |                                   |                |                    | interval. This can be much faster for large datasets                 and         |            |                                                                                            |
    |                                   |                |                    | solution intervals.                                                              |            |                                                                                            |
    +-----------------------------------+----------------+--------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.refant          | ['int', 'str'] | 0                  | Reference antenna.                 Currently only activated for                  | False      |                                                                                            |
    |                                   |                |                    | gain_substitution solver                                                         |            |                                                                                            |
    +-----------------------------------+----------------+--------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.niter           | int            | 50                 | Number of solver iterations.                                                     | False      |                                                                                            |
    +-----------------------------------+----------------+--------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.phase_only      | bool           | False              | Solve only for the phases. This can be set                 to ``True`` when      | False      |                                                                                            |
    |                                   |                |                    | solver is "gain_substitution",                 otherwise it must be ``False``.   |            |                                                                                            |
    +-----------------------------------+----------------+--------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.tol             | float          | 0.001              | Iteration stops when the fractional change                 in the gain solution  | False      |                                                                                            |
    |                                   |                |                    | is below this tolerance.                                                         |            |                                                                                            |
    +-----------------------------------+----------------+--------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.crosspol        | bool           | False              | Do solutions including cross polarisations                 i.e. XY, YX or RL,    | False      |                                                                                            |
    |                                   |                |                    | LR.                 Only used by "gain_substitution" solver.                     |            |                                                                                            |
    +-----------------------------------+----------------+--------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.normalise_gains | str            | ``null``           | Normalises the gains.                 Only available when solver is              | True       | ['``null``', 'mean', 'median']                                                             |
    |                                   |                |                    | "gain_substitution".                 Possible types of normalization are:        |            |                                                                                            |
    |                                   |                |                    | "mean", "median".                 To perform no normalization, set this to       |            |                                                                                            |
    |                                   |                |                    | ``null``.                                                                        |            |                                                                                            |
    +-----------------------------------+----------------+--------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.timeslice       | float          | ``null``           | Defines time scale over which each gain solution                 is valid. This  | True       |                                                                                            |
    |                                   |                |                    | is used to define time axis of the GainTable.                 This parameter is  |            |                                                                                            |
    |                                   |                |                    | interpreted as follows,                  float: this is a custom time interval   |            |                                                                                            |
    |                                   |                |                    | in seconds.                 Input timestamps are grouped by intervals of this    |            |                                                                                            |
    |                                   |                |                    | duration,                 and said groups are separately averaged to produce     |            |                                                                                            |
    |                                   |                |                    | the output time axis.                  ``None``: match the time resolution of    |            |                                                                                            |
    |                                   |                |                    | the input, i.e. copy                 the time axis of the input Visibility       |            |                                                                                            |
    +-----------------------------------+----------------+--------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | export_gaintable                  | bool           | False              | Export intermediate gain solutions.                                              | False      |                                                                                            |
    +-----------------------------------+----------------+--------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+


delay_calibration
=================

    Performs delay calibration

Parameters
----------

..  table::
    :width: 100%
    :widths: 15, 10, 10, 45, 10, 10

    +------------------------+--------+-----------+-------------------------------------+------------+------------------+
    | Param                  | Type   | Default   | Description                         | Nullable   | Allowed values   |
    +========================+========+===========+=====================================+============+==================+
    | oversample             | int    | 16        | Oversample rate                     | True       |                  |
    +------------------------+--------+-----------+-------------------------------------+------------+------------------+
    | plot_config.plot_table | bool   | False     | Plot the generated gaintable        | True       |                  |
    +------------------------+--------+-----------+-------------------------------------+------------+------------------+
    | plot_config.fixed_axis | bool   | False     | Limit amplitude axis to [0-1]       | True       |                  |
    +------------------------+--------+-----------+-------------------------------------+------------+------------------+
    | export_gaintable       | bool   | False     | Export intermediate gain solutions. | False      |                  |
    +------------------------+--------+-----------+-------------------------------------+------------+------------------+


smooth_gain_solution
====================

    Smooth the gain solution.

Parameters
----------

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
===================

    Apply gaintable and export visibilities.

Parameters
----------

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
=================

    Export gain table solutions to a file.

Parameters
----------

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




Target Calibration Stages
*************************

This section describes the stages used in the Target Calibration pipeline.

target_load_data
================

    This stage loads the target visibility data from either (in order of
    preference):

    1. An existing dataset stored as a zarr file inside the 'cache_directory'.
    2. From input MSv2 measurement set. Here it will create an intemediate
       zarr file with chunks along frequency and time, then use it as input
       to the pipeline. This zarr dataset will be stored in 'cache_directory'
       for later use.

Parameters
----------

..  table::
    :width: 100%
    :widths: 15, 10, 10, 45, 10, 10

    +---------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------------------------------+
    | Param               | Type   | Default   | Description                                                                      | Nullable   | Allowed values                           |
    +=====================+========+===========+==================================================================================+============+==========================================+
    | nchannels_per_chunk | int    | 32        | Number of frequency channels per chunk in the             written zarr file.     | False      |                                          |
    +---------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------------------------------+
    | ntimes_per_ms_chunk | int    | 5         | Number of time slots to include in each chunk             while reading from     | False      |                                          |
    |                     |        |           | measurement set and writing in zarr file.             This is also the size of   |            |                                          |
    |                     |        |           | time chunk used across the pipeline.                                             |            |                                          |
    +---------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------------------------------+
    | cache_directory     | str    | ``null``  | Cache directory containing previously stored             visibility datasets as  | True       |                                          |
    |                     |        |           | zarr files. The directory should contain             a subdirectory with same    |            |                                          |
    |                     |        |           | name as the input target ms file name,             which internally contains the |            |                                          |
    |                     |        |           | zarr and pickle files.             If None, the input ms will be converted to    |            |                                          |
    |                     |        |           | zarr file,             and this zarr file will be stored in a new 'cache'        |            |                                          |
    |                     |        |           | subdirectory under the provided output directory.                                |            |                                          |
    +---------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------------------------------+
    | ack                 | bool   | False     | Ask casacore to acknowledge each table operation                                 | False      |                                          |
    +---------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------------------------------+
    | datacolumn          | str    | DATA      | MS data column to read visibility data from.                                     | False      | ['DATA', 'CORRECTED_DATA', 'MODEL_DATA'] |
    +---------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------------------------------+
    | field_id            | int    | 0         | Field ID of the data in measurement set                                          | False      |                                          |
    +---------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------------------------------+
    | data_desc_id        | int    | 0         | Data Description ID of the data in measurement set                               | False      |                                          |
    +---------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------------------------------+


predict_vis
===========

    Predict model visibilities using a local sky model.

Parameters
----------

..  table::
    :width: 100%
    :widths: 15, 10, 10, 45, 10, 10

    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | Param                    | Type   | Default   | Description                                                                      | Nullable   | Allowed values   |
    +==========================+========+===========+==================================================================================+============+==================+
    | beam_type                | str    | everybeam | Type of beam model to use.                                                       | True       |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | normalise_at_beam_centre | bool   | True      | If true, before running calibration, multiply vis             and model vis by   | True       |                  |
    |                          |        |           | the inverse of the beam response in the             beam pointing direction.     |            |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | eb_ms                    | str    | ``null``  | If beam_type is "everybeam" but input ms does             not have all of the    | True       |                  |
    |                          |        |           | metadata required by everybeam, this parameter             is used to specify a  |            |                  |
    |                          |        |           | separate dataset to use when setting up             the beam models.             |            |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | eb_coeffs                | str    | ``null``  | Everybeam coeffs datadir containing beam             coefficients. Required if   | True       |                  |
    |                          |        |           | beam_type is 'everybeam'.                                                        |            |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | gleamfile                | str    | ``null``  | Specifies the location of gleam catalogue             file gleamegc.dat          | True       |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | lsm_csv_path             | str    | ``null``  | Specifies the location of CSV file containing the             sky model. The CSV | True       |                  |
    |                          |        |           | file should be in OSKAR CSV format.                                              |            |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | fov                      | float  | 10.0      | Specifies the width of the cone used when             searching for compoents,   | True       |                  |
    |                          |        |           | in units of degrees.                                                             |            |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | flux_limit               | float  | 1.0       | Specifies the flux density limit used when             searching for compoents,  | True       |                  |
    |                          |        |           | in units of Jy.                                                                  |            |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | alpha0                   | float  | -0.78     | Nominal alpha value to use when fitted data             are unspecified..        | True       |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+


complex_gain_calibration
========================

    Performs Complex Gain Calibration

Parameters
----------

..  table::
    :width: 100%
    :widths: 15, 10, 10, 45, 10, 10

    +-----------------------------------+----------------+-------------------+---------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | Param                             | Type           | Default           | Description                                                                     | Nullable   | Allowed values                                                                             |
    +===================================+================+===================+=================================================================================+============+============================================================================================+
    | run_solver_config.solver          | str            | gain_substitution | Calibration algorithm to use.                 (default="gain_substitution")     | True       | ['gain_substitution', 'jones_substitution', 'normal_equations', 'normal_equations_presum'] |
    |                                   |                |                   | Options are:                 "gain_substitution" - original substitution        |            |                                                                                            |
    |                                   |                |                   | algorithm                 with separate solutions for each polarisation term.   |            |                                                                                            |
    |                                   |                |                   | "jones_substitution" - solve antenna-based Jones matrices                 as a  |            |                                                                                            |
    |                                   |                |                   | whole, with independent updates within each iteration.                          |            |                                                                                            |
    |                                   |                |                   | "normal_equations" - solve normal equations within                 each         |            |                                                                                            |
    |                                   |                |                   | iteration formed from linearisation with respect to                 antenna-    |            |                                                                                            |
    |                                   |                |                   | based gain and leakage terms.                 "normal_equations_presum" - same  |            |                                                                                            |
    |                                   |                |                   | as normal_equations                 option but with an initial accumulation of  |            |                                                                                            |
    |                                   |                |                   | visibility                 products over time and frequency for each solution   |            |                                                                                            |
    |                                   |                |                   | interval. This can be much faster for large datasets                 and        |            |                                                                                            |
    |                                   |                |                   | solution intervals.                                                             |            |                                                                                            |
    +-----------------------------------+----------------+-------------------+---------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.refant          | ['int', 'str'] | 0                 | Reference antenna.                 Currently only activated for                 | False      |                                                                                            |
    |                                   |                |                   | gain_substitution solver                                                        |            |                                                                                            |
    +-----------------------------------+----------------+-------------------+---------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.niter           | int            | 50                | Number of solver iterations.                                                    | False      |                                                                                            |
    +-----------------------------------+----------------+-------------------+---------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.tol             | float          | 1e-06             | Iteration stops when the fractional change                 in the gain solution | False      |                                                                                            |
    |                                   |                |                   | is below this tolerance.                                                        |            |                                                                                            |
    +-----------------------------------+----------------+-------------------+---------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.crosspol        | bool           | False             | Do solutions including cross polarisations                 i.e. XY, YX or RL,   | False      |                                                                                            |
    |                                   |                |                   | LR.                 Only used by "gain_substitution" solver.                    |            |                                                                                            |
    +-----------------------------------+----------------+-------------------+---------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.normalise_gains | str            | ``null``          | Normalises the gains.                 Only available when solver is             | True       | ['``null``', 'mean', 'median']                                                             |
    |                                   |                |                   | "gain_substitution".                 Possible types of normalization are:       |            |                                                                                            |
    |                                   |                |                   | "mean", "median".                 To perform no normalization, set this to      |            |                                                                                            |
    |                                   |                |                   | ``null``.                                                                       |            |                                                                                            |
    +-----------------------------------+----------------+-------------------+---------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.timeslice       | float          | ``null``          | Defines time scale over which each gain solution                 is valid. This | True       |                                                                                            |
    |                                   |                |                   | is used to define time axis of the GainTable.                 This parameter is |            |                                                                                            |
    |                                   |                |                   | interpreted as follows,                  float: this is a custom time interval  |            |                                                                                            |
    |                                   |                |                   | in seconds.                 Input timestamps are grouped by intervals of this   |            |                                                                                            |
    |                                   |                |                   | duration,                 and said groups are separately averaged to produce    |            |                                                                                            |
    |                                   |                |                   | the output time axis.                  ``None``: match the time resolution of   |            |                                                                                            |
    |                                   |                |                   | the input, i.e. copy                 the time axis of the input Visibility      |            |                                                                                            |
    +-----------------------------------+----------------+-------------------+---------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | plot_config.plot_table            | bool           | False             | Plot the generated gaintable                                                    | False      |                                                                                            |
    +-----------------------------------+----------------+-------------------+---------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | plot_config.fixed_axis            | bool           | False             | Limit amplitude axis to [0-1]                                                   | False      |                                                                                            |
    +-----------------------------------+----------------+-------------------+---------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | visibility_key                    | str            | vis               | Visibility data to be used for calibration.                                     | True       | ['vis', 'corrected_vis']                                                                   |
    +-----------------------------------+----------------+-------------------+---------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | export_gaintable                  | bool           | False             | Export intermediate gain solutions.                                             | False      |                                                                                            |
    +-----------------------------------+----------------+-------------------+---------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+


export_gain_table
=================

    Export gain table solutions to a file.

Parameters
----------

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




Target Ionospheric Calibration Stages
*************************

This section describes the stages used in the Target Ionospheric pipeline.

target_load_data
================

    This stage loads the target visibility data from either (in order of
    preference):

    1. An existing dataset stored as a zarr file inside the 'cache_directory'.
    2. From input MSv2 measurement set. Here it will create an intemediate
       zarr file with chunks along frequency and time, then use it as input
       to the pipeline. This zarr dataset will be stored in 'cache_directory'
       for later use.

Parameters
----------

..  table::
    :width: 100%
    :widths: 15, 10, 10, 45, 10, 10

    +---------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------------------------------+
    | Param               | Type   | Default   | Description                                                                      | Nullable   | Allowed values                           |
    +=====================+========+===========+==================================================================================+============+==========================================+
    | nchannels_per_chunk | int    | 32        | Number of frequency channels per chunk in the             written zarr file.     | False      |                                          |
    +---------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------------------------------+
    | ntimes_per_ms_chunk | int    | 5         | Number of time slots to include in each chunk             while reading from     | False      |                                          |
    |                     |        |           | measurement set and writing in zarr file.             This is also the size of   |            |                                          |
    |                     |        |           | time chunk used across the pipeline.                                             |            |                                          |
    +---------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------------------------------+
    | cache_directory     | str    | ``null``  | Cache directory containing previously stored             visibility datasets as  | True       |                                          |
    |                     |        |           | zarr files. The directory should contain             a subdirectory with same    |            |                                          |
    |                     |        |           | name as the input target ms file name,             which internally contains the |            |                                          |
    |                     |        |           | zarr and pickle files.             If None, the input ms will be converted to    |            |                                          |
    |                     |        |           | zarr file,             and this zarr file will be stored in a new 'cache'        |            |                                          |
    |                     |        |           | subdirectory under the provided output directory.                                |            |                                          |
    +---------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------------------------------+
    | ack                 | bool   | False     | Ask casacore to acknowledge each table operation                                 | False      |                                          |
    +---------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------------------------------+
    | datacolumn          | str    | DATA      | MS data column to read visibility data from.                                     | False      | ['DATA', 'CORRECTED_DATA', 'MODEL_DATA'] |
    +---------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------------------------------+
    | field_id            | int    | 0         | Field ID of the data in measurement set                                          | False      |                                          |
    +---------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------------------------------+
    | data_desc_id        | int    | 0         | Data Description ID of the data in measurement set                               | False      |                                          |
    +---------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------------------------------+


predict_vis
===========

    Predict model visibilities using a local sky model.

Parameters
----------

..  table::
    :width: 100%
    :widths: 15, 10, 10, 45, 10, 10

    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | Param                    | Type   | Default   | Description                                                                      | Nullable   | Allowed values   |
    +==========================+========+===========+==================================================================================+============+==================+
    | beam_type                | str    | everybeam | Type of beam model to use.                                                       | True       |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | normalise_at_beam_centre | bool   | True      | If true, before running calibration, multiply vis             and model vis by   | True       |                  |
    |                          |        |           | the inverse of the beam response in the             beam pointing direction.     |            |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | eb_ms                    | str    | ``null``  | If beam_type is "everybeam" but input ms does             not have all of the    | True       |                  |
    |                          |        |           | metadata required by everybeam, this parameter             is used to specify a  |            |                  |
    |                          |        |           | separate dataset to use when setting up             the beam models.             |            |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | eb_coeffs                | str    | ``null``  | Everybeam coeffs datadir containing beam             coefficients. Required if   | True       |                  |
    |                          |        |           | beam_type is 'everybeam'.                                                        |            |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | gleamfile                | str    | ``null``  | Specifies the location of gleam catalogue             file gleamegc.dat          | True       |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | lsm_csv_path             | str    | ``null``  | Specifies the location of CSV file containing the             sky model. The CSV | True       |                  |
    |                          |        |           | file should be in OSKAR CSV format.                                              |            |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | fov                      | float  | 10.0      | Specifies the width of the cone used when             searching for compoents,   | True       |                  |
    |                          |        |           | in units of degrees.                                                             |            |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | flux_limit               | float  | 1.0       | Specifies the flux density limit used when             searching for compoents,  | True       |                  |
    |                          |        |           | in units of Jy.                                                                  |            |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | alpha0                   | float  | -0.78     | Nominal alpha value to use when fitted data             are unspecified..        | True       |                  |
    +--------------------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+


ionospheric_delay
=================

    Calculates and applies ionospheric delay corrections to visibility data.

    This function uses an IonosphericSolver to model phase screens based on
    the difference between observed visibilities and model visibilities. It
    derives a gain table representing these phase corrections and applies it
    to the visibility data. The resulting gain table can be optionally
    exported to an H5parm file.

Parameters
----------

..  table::
    :width: 100%
    :widths: 15, 10, 10, 45, 10, 10

    +-----------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | Param           | Type   | Default   | Description                                                                      | Nullable   | Allowed values   |
    +=================+========+===========+==================================================================================+============+==================+
    | timeslice       | float  | 3.0       | Defines time scale over which each gain solution                 is valid. This  | True       |                  |
    |                 |        |           | is used to define time axis of the GainTable.                 This parameter is  |            |                  |
    |                 |        |           | interpreted as follows,                  float: this is a custom time interval   |            |                  |
    |                 |        |           | in seconds.                 Input timestamps are grouped by intervals of this    |            |                  |
    |                 |        |           | duration,                 and said groups are separately averaged to produce     |            |                  |
    |                 |        |           | the output time axis.                  ``None``: match the time resolution of    |            |                  |
    |                 |        |           | the input, i.e. copy                 the time axis of the input Visibility       |            |                  |
    +-----------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | cluster_indexes | list   | ``null``  | Array of integers assigning each antenna to a cluster. If None, all antennas are | True       |                  |
    |                 |        |           | treated as a single cluster                                                      |            |                  |
    +-----------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | block_diagonal  | bool   | True      | If True, solve for all clusters simultaneously assuming a block-diagonal system. | False      |                  |
    |                 |        |           | If False, solve for each cluster sequentially                                    |            |                  |
    +-----------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | niter           | int    | 10        | Number of solver iterations.                                                     | False      |                  |
    +-----------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | tol             | float  | 1e-06     | Iteration stops when the fractional change             in the gain solution is   | False      |                  |
    |                 |        |           | below this tolerance.                                                            |            |                  |
    +-----------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | zernike_limit   | int    | ``null``  | The maximum order of Zernike polynomials to use for the screen model.            | True       |                  |
    +-----------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+
    | plot_table      | bool   | False     | Plot all station Phase vs Frequency                                              | False      |                  |
    +-----------------+--------+-----------+----------------------------------------------------------------------------------+------------+------------------+


export_gain_table
=================

    Export gain table solutions to a file.

Parameters
----------

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





