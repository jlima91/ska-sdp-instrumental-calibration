Stages and configurations
#########################

.. This page is generated using docs/generate_config.py

The descriptions of each stage are copied from the docstrings of stages.
Refer to the `API page for stages <api/ska_sdp_spectral_line_imaging.stages.html>`_

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
    | selected_sources | list   | None      | Sources to select                                                              | True       |                                          |
    +------------------+--------+-----------+--------------------------------------------------------------------------------+------------+------------------------------------------+
    | selected_dds     | list   | None      | Data descriptors to select                                                     | True       |                                          |
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

    +------------------+--------+-----------+----------------------------------------------------------------------------------+-------------------------------------+------------------+
    | Param            | Type   | Default   | Description                                                                      | Nullable                            | Allowed values   |
    +==================+========+===========+==================================================================================+=====================================+==================+
    | beam_type        | str    | everybeam | Type of beam model to use. Default is 'everybeam'                                | True                                |                  |
    +------------------+--------+-----------+----------------------------------------------------------------------------------+-------------------------------------+------------------+
    | eb_ms            | str    | Param     | Measurement set used to initialise the everybeam telescope. Required if          | True                                |                  |
    |                  |        | ms_name   | bbeam_type is 'everybeam' and the main ms, ms_name, is missing beam information. |                                     |                  |
    +------------------+--------+-----------+----------------------------------------------------------------------------------+-------------------------------------+------------------+
    | eb_coeffs        | str    | None      | Everybeam coeffs datadir containing beam             coefficients. Required if   | True                                |                  |
    |                  |        |           | bbeam_type is 'everybeam'.                                                       |                                     |                  |
    +------------------+--------+-----------+----------------------------------------------------------------------------------+-------------------------------------+------------------+
    | lsm              | list   | None      | Optional list of lsm Component objects to use as the local sky model.            | True                                |                  |
    +------------------+--------+-----------+----------------------------------------------------------------------------------+-------------------------------------+------------------+
    | gleamfile        | str    | None      | Specifies the location of gleam catalogue file gleamegc.dat. If lsm is None, the | True                                |                  |
    |                  |        |           | sky model must be specified by either a gleamegc catalogue file or a csv file.   |                                     |                  |
    +------------------+--------+-----------+----------------------------------------------------------------------------------+-------------------------------------+------------------+
    | csvfile          | str    | None      | Specifies the location of a csv sky component list file. If lsm is None, the sky | True                                |                  |
    |                  |        |           | model must be specified by either a gleamegc catalogue file or a csv file.       |                                     |                  |
    +------------------+--------+-----------+----------------------------------------------------------------------------------+-------------------------------------+------------------+
    | lsm_csv_path     | str    | None      | Specifies the location of CSV file for custom             components             | True                                |                  |
    +------------------+--------+-----------+----------------------------------------------------------------------------------+-------------------------------------+------------------+
    | fov              | float  | 10.0      | Specifies the width of the cone used when             searching for compoents,   | True                                |                  |
    |                  |        |           | in units of degrees. Default: 10.                                                |                                     |                  |
    +------------------+--------+-----------+----------------------------------------------------------------------------------+-------------------------------------+------------------+
    | flux_limit       | float  | 1.0       | Specifies the flux density limit used when             searching for compoents,  | True                                |                  |
    |                  |        |           | in units of Jy. Defaults to 1                                                    |                                     |                  |
    +------------------+--------+-----------+----------------------------------------------------------------------------------+-------------------------------------+------------------+
    | export_model_vis | bool   | False     | None                                                                             | Export predicted model visibilities |                  |
    +------------------+--------+-----------+----------------------------------------------------------------------------------+-------------------------------------+------------------+


bandpass_calibration
********************

    Performs Bandpass Calibration

Parameters
==========

..  table::
    :width: 100%
    :widths: 15, 10, 10, 45, 10, 10

    +--------------------------+--------+-------------------+---------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | Param                    | Type   | Default           | Description                                                               | Nullable   | Allowed values                                                                             |
    +==========================+========+===================+===========================================================================+============+============================================================================================+
    | run_solver_config.solver | str    | gain_substitution | Solver type to use. Currently any solver                 type accepted by | True       | ['gain_substitution', 'jones_substitution', 'normal_equations', 'normal_equations_presum'] |
    |                          |        |                   | solve_gaintable.                 Default is 'gain_substitution'.          |            |                                                                                            |
    +--------------------------+--------+-------------------+---------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.refant | int    | 0                 | Reference antenna (defaults to 0).                                        | True       |                                                                                            |
    +--------------------------+--------+-------------------+---------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.niter  | int    | 50                | Number of solver iterations (defaults to 50)                              | True       |                                                                                            |
    +--------------------------+--------+-------------------+---------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | plot_config.plot_table   | bool   | False             | Plot the generated gaintable                                              | True       |                                                                                            |
    +--------------------------+--------+-------------------+---------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | plot_config.fixed_axis   | bool   | False             | Limit amplitude axis to [0-1]                                             | True       |                                                                                            |
    +--------------------------+--------+-------------------+---------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | flagging                 | bool   | False             | Run RFI flagging                                                          | True       |                                                                                            |
    +--------------------------+--------+-------------------+---------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+


generate_channel_rm
*******************

    Generates channel rotation measures

Parameters
==========

..  table::
    :width: 100%
    :widths: 15, 10, 10, 45, 10, 10

    +--------------------------+--------+------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | Param                    | Type   | Default          | Description                                                                      | Nullable   | Allowed values                                                                             |
    +==========================+========+==================+==================================================================================+============+============================================================================================+
    | fchunk                   | int    | -1               | Number of frequency channels per chunk.             If set to -1, use fchunk     | True       |                                                                                            |
    |                          |        |                  | value from load_data                                                             |            |                                                                                            |
    +--------------------------+--------+------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | peak_threshold           | float  | 0.5              | Height of peak in the RM spectrum required             for a rotation detection. | True       |                                                                                            |
    +--------------------------+--------+------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | plot_table               | bool   | False            | Plot the generated gain table                                                    | True       |                                                                                            |
    +--------------------------+--------+------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.solver | str    | normal_equations | Solver type to use. Currently any solver                 type accepted by        | True       | ['gain_substitution', 'jones_substitution', 'normal_equations', 'normal_equations_presum'] |
    |                          |        |                  | solve_gaintable.                 Default is 'normal_equations'.                  |            |                                                                                            |
    +--------------------------+--------+------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.refant | int    | 0                | Reference antenna (defaults to 0).                                               | True       |                                                                                            |
    +--------------------------+--------+------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.niter  | int    | 50               | Number of solver iterations (defaults to 50)                                     | True       |                                                                                            |
    +--------------------------+--------+------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+


delay_calibration
*****************

    Performs delay calibration

Parameters
==========

..  table::
    :width: 100%
    :widths: 15, 10, 10, 45, 10, 10

    +------------------------+--------+-----------+-------------------------------+------------+------------------+
    | Param                  | Type   | Default   | Description                   | Nullable   | Allowed values   |
    +========================+========+===========+===============================+============+==================+
    | oversample             | int    | 16        | Oversample rate               | True       |                  |
    +------------------------+--------+-----------+-------------------------------+------------+------------------+
    | plot_config.plot_table | bool   | False     | Plot the generated gaintable  | True       |                  |
    +------------------------+--------+-----------+-------------------------------+------------+------------------+
    | plot_config.fixed_axis | bool   | False     | Limit amplitude axis to [0-1] | True       |                  |
    +------------------------+--------+-----------+-------------------------------+------------+------------------+


export_gain_table
*****************

    Export gain table solutions to a file.

Parameters
==========

..  table::
    :width: 100%
    :widths: 15, 10, 10, 45, 10, 10

    +---------------+--------+-----------+----------------------------------------+------------+--------------------+
    | Param         | Type   | Default   | Description                            | Nullable   | Allowed values     |
    +===============+========+===========+========================================+============+====================+
    | file_name     | str    | gaintable | Gain table file name without extension | True       |                    |
    +---------------+--------+-----------+----------------------------------------+------------+--------------------+
    | export_format | str    | h5parm    | Export file format                     | True       | ['h5parm', 'hdf5'] |
    +---------------+--------+-----------+----------------------------------------+------------+--------------------+


Stages and configurations
#########################

.. This page is generated using docs/generate_config.py

The descriptions of each stage are copied from the docstrings of stages.
Refer to the `API page for stages <api/ska_sdp_spectral_line_imaging.stages.html>`_

Each stage has parameters, which are defined in the YAML config file passed to the pipeline.


load_data
*********

    Load the Measurement Set data.

Parameters
==========

..  table::
    :width: 100%
    :widths: 15, 10, 10, 45, 10, 10

    +---------+--------+-----------+----------------------------------------+------------+------------------+
    | Param   | Type   | Default   | Description                            | Nullable   | Allowed values   |
    +=========+========+===========+========================================+============+==================+
    | fchunk  | int    | 32        | Number of frequency channels per chunk | True       |                  |
    +---------+--------+-----------+----------------------------------------+------------+------------------+


predict_vis
***********

    Predict model visibilities using a local sky model.

Parameters
==========

..  table::
    :width: 100%
    :widths: 15, 10, 10, 45, 10, 10

    +------------------+--------+-----------+----------------------------------------------------------------------------------+-------------------------------------+------------------+
    | Param            | Type   | Default   | Description                                                                      | Nullable                            | Allowed values   |
    +==================+========+===========+==================================================================================+=====================================+==================+
    | beam_type        | str    | everybeam | Type of beam model to use. Default is 'everybeam'                                | True                                |                  |
    +------------------+--------+-----------+----------------------------------------------------------------------------------+-------------------------------------+------------------+
    | eb_ms            | str    | None      | Measurement set need to initialise the everybeam             telescope. Required | True                                |                  |
    |                  |        |           | if bbeam_type is 'everybeam'.                                                    |                                     |                  |
    +------------------+--------+-----------+----------------------------------------------------------------------------------+-------------------------------------+------------------+
    | eb_coeffs        | str    | None      | Everybeam coeffs datadir containing beam             coefficients. Required if   | True                                |                  |
    |                  |        |           | bbeam_type is 'everybeam'.                                                       |                                     |                  |
    +------------------+--------+-----------+----------------------------------------------------------------------------------+-------------------------------------+------------------+
    | gleamfile        | str    | None      | Specifies the location of gleam catalogue             file gleamegc.dat          | True                                |                  |
    +------------------+--------+-----------+----------------------------------------------------------------------------------+-------------------------------------+------------------+
    | fov              | float  | 10.0      | Specifies the width of the cone used when             searching for compoents,   | True                                |                  |
    |                  |        |           | in units of degrees. Default: 10.                                                |                                     |                  |
    +------------------+--------+-----------+----------------------------------------------------------------------------------+-------------------------------------+------------------+
    | flux_limit       | float  | 1.0       | Specifies the flux density limit used when             searching for compoents,  | True                                |                  |
    |                  |        |           | in units of Jy. Defaults to 1                                                    |                                     |                  |
    +------------------+--------+-----------+----------------------------------------------------------------------------------+-------------------------------------+------------------+
    | alpha0           | float  | -0.78     | Nominal alpha value to use when fitted data             are unspecified. Default | True                                |                  |
    |                  |        |           | is -0.78.                                                                        |                                     |                  |
    +------------------+--------+-----------+----------------------------------------------------------------------------------+-------------------------------------+------------------+
    | reset_vis        | bool   | False     | Whether or not to set visibilities to zero before             accumulating       | True                                |                  |
    |                  |        |           | components. Default is False.                                                    |                                     |                  |
    +------------------+--------+-----------+----------------------------------------------------------------------------------+-------------------------------------+------------------+
    | export_model_vis | bool   | False     | None                                                                             | Export predicted model visibilities |                  |
    +------------------+--------+-----------+----------------------------------------------------------------------------------+-------------------------------------+------------------+


bandpass_calibration
********************

    Performs Bandpass Calibration

Parameters
==========

..  table::
    :width: 100%
    :widths: 15, 10, 10, 45, 10, 10

    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | Param                             | Type   | Default           | Description                                                                      | Nullable   | Allowed values                                                                             |
    +===================================+========+===================+==================================================================================+============+============================================================================================+
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
    | run_solver_config.refant          | int    | 0                 | Reference antenna (default 0).                 Currently only activated for      | True       |                                                                                            |
    |                                   |        |                   | gain_substitution solver                                                         |            |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.niter           | int    | 50                | Number of solver iterations (defaults to 50)                                     | True       |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.phase_only      | bool   | False             | Solve only for the phases. default=True when                                     | True       |                                                                                            |
    |                                   |        |                   | solver="gain_substitution", otherwise it must be False.                          |            |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.tol             | float  | 1e-06             | Iteration stops when the fractional change                 in the gain solution  | True       |                                                                                            |
    |                                   |        |                   | is below this tolerance (default=1e-6)                                           |            |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.crosspol        | bool   | False             | Do solutions including cross polarisations                 i.e. XY, YX or RL,    | True       |                                                                                            |
    |                                   |        |                   | LR. Only used by gain_substitution.                                              |            |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.normalise_gains | str    | mean              | Normalises the gains (default="mean").                 Options are None, "mean", | True       | [None, 'mean', 'median']                                                                   |
    |                                   |        |                   | "median".                 None means no normalization.                 Only      |            |                                                                                            |
    |                                   |        |                   | available with gain_substitution.                                                |            |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.jones_type      | str    | T                 | Type of calibration matrix T or G or B.                 The frequency axis of    | True       | ['T', 'G', 'B']                                                                            |
    |                                   |        |                   | the output GainTable                 depends on the value provided:              |            |                                                                                            |
    |                                   |        |                   | "B": the output frequency axis is the same as                 that of the input  |            |                                                                                            |
    |                                   |        |                   | Visibility.                 "T" or "G": the solution is assumed to be            |            |                                                                                            |
    |                                   |        |                   | frequency-independent, and the frequency axis of the                 output      |            |                                                                                            |
    |                                   |        |                   | contains a single value: the average frequency                 of the input      |            |                                                                                            |
    |                                   |        |                   | Visibility's channels.                                                           |            |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.timeslice       | float  | None              | Defines time scale over which each gain solution                 is valid. This  | True       |                                                                                            |
    |                                   |        |                   | is used to define time axis of the GainTable.                 This parameter is  |            |                                                                                            |
    |                                   |        |                   | interpreted as follows,                 float: this is a custom time interval in |            |                                                                                            |
    |                                   |        |                   | seconds.                 Input timestamps are grouped by intervals of this       |            |                                                                                            |
    |                                   |        |                   | duration,                 and said groups are separately averaged to produce     |            |                                                                                            |
    |                                   |        |                   | the output time axis.                 None: match the time resolution of the     |            |                                                                                            |
    |                                   |        |                   | input, i.e. copy                 the time axis of the input Visibility           |            |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | plot_config.plot_table            | bool   | False             | Plot the generated gaintable                                                     | True       |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | plot_config.fixed_axis            | bool   | False             | Limit amplitude axis to [0-1]                                                    | True       |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | flagging                          | bool   | False             | Run RFI flagging                                                                 | True       |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+


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
    | run_solver_config.refant          | int    | 0                 | Reference antenna (default 0).                 Currently only activated for      | True       |                                                                                            |
    |                                   |        |                   | gain_substitution solver                                                         |            |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.niter           | int    | 50                | Number of solver iterations (defaults to 50)                                     | True       |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.phase_only      | bool   | False             | Solve only for the phases. default=True when                                     | True       |                                                                                            |
    |                                   |        |                   | solver="gain_substitution", otherwise it must be False.                          |            |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.tol             | float  | 1e-06             | Iteration stops when the fractional change                 in the gain solution  | True       |                                                                                            |
    |                                   |        |                   | is below this tolerance (default=1e-6)                                           |            |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.crosspol        | bool   | False             | Do solutions including cross polarisations                 i.e. XY, YX or RL,    | True       |                                                                                            |
    |                                   |        |                   | LR. Only used by gain_substitution.                                              |            |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.normalise_gains | str    | mean              | Normalises the gains (default="mean").                 Options are None, "mean", | True       | [None, 'mean', 'median']                                                                   |
    |                                   |        |                   | "median".                 None means no normalization.                 Only      |            |                                                                                            |
    |                                   |        |                   | available with gain_substitution.                                                |            |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.jones_type      | str    | T                 | Type of calibration matrix T or G or B.                 The frequency axis of    | True       | ['T', 'G', 'B']                                                                            |
    |                                   |        |                   | the output GainTable                 depends on the value provided:              |            |                                                                                            |
    |                                   |        |                   | "B": the output frequency axis is the same as                 that of the input  |            |                                                                                            |
    |                                   |        |                   | Visibility.                 "T" or "G": the solution is assumed to be            |            |                                                                                            |
    |                                   |        |                   | frequency-independent, and the frequency axis of the                 output      |            |                                                                                            |
    |                                   |        |                   | contains a single value: the average frequency                 of the input      |            |                                                                                            |
    |                                   |        |                   | Visibility's channels.                                                           |            |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+
    | run_solver_config.timeslice       | float  | None              | Defines time scale over which each gain solution                 is valid. This  | True       |                                                                                            |
    |                                   |        |                   | is used to define time axis of the GainTable.                 This parameter is  |            |                                                                                            |
    |                                   |        |                   | interpreted as follows,                 float: this is a custom time interval in |            |                                                                                            |
    |                                   |        |                   | seconds.                 Input timestamps are grouped by intervals of this       |            |                                                                                            |
    |                                   |        |                   | duration,                 and said groups are separately averaged to produce     |            |                                                                                            |
    |                                   |        |                   | the output time axis.                 None: match the time resolution of the     |            |                                                                                            |
    |                                   |        |                   | input, i.e. copy                 the time axis of the input Visibility           |            |                                                                                            |
    +-----------------------------------+--------+-------------------+----------------------------------------------------------------------------------+------------+--------------------------------------------------------------------------------------------+


delay_calibration
*****************

    Performs delay calibration

Parameters
==========

..  table::
    :width: 100%
    :widths: 15, 10, 10, 45, 10, 10

    +------------------------+--------+-----------+-------------------------------+------------+------------------+
    | Param                  | Type   | Default   | Description                   | Nullable   | Allowed values   |
    +========================+========+===========+===============================+============+==================+
    | oversample             | int    | 16        | Oversample rate               | True       |                  |
    +------------------------+--------+-----------+-------------------------------+------------+------------------+
    | plot_config.plot_table | bool   | False     | Plot the generated gaintable  | True       |                  |
    +------------------------+--------+-----------+-------------------------------+------------+------------------+
    | plot_config.fixed_axis | bool   | False     | Limit amplitude axis to [0-1] | True       |                  |
    +------------------------+--------+-----------+-------------------------------+------------+------------------+


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
