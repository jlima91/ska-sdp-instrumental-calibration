Running the Pipeline
====================

The INST pipelines use `dask <https://docs.dask.org>`_ for their computations. Depending on the options passed to the ``run`` subcommand, the computation can run in one of three ways:

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Mode
     - Options
     - When to use
   * - Local threads (default)
     - none
     - Small datasets, quick tests, debugging.
   * - Pipeline-created cluster
     - ``--create-dask-cluster``
     - Distributed runs on a single machine, or across the nodes of a SLURM allocation, without managing a cluster yourself.
   * - Existing dask cluster
     - ``--dask-scheduler <address>``
     - When you want full control over the scheduler and workers, or want to reuse a long-running cluster.

The examples below assume that the pipeline is installed and that a configuration file has been generated using the ``install-config`` subcommand (see :doc:`Getting Started <README>`).

The examples use ``ska-sdp-instrumental-calibration``. The ``run`` subcommands of ``ska-sdp-instrumental-target-calibration``, ``ska-sdp-instrumental-target-ionospheric``, and the ``experimental`` subcommand accept the same options.

For regular runs, the repository also has helper scripts which wrap these commands. See `Using the helper scripts`_.

Run with local threads
----------------------

When neither ``--dask-scheduler`` nor ``--create-dask-cluster`` is passed, no dask cluster is used. Dask computes the task graph locally, in the pipeline process, using a thread pool.

.. code-block:: bash

   ska-sdp-instrumental-calibration run \
       --config config.yml \
       /path/to/data.ms

Let the pipeline create a dask cluster
--------------------------------------

With ``--create-dask-cluster``, the pipeline starts its own dask cluster before the run, and shuts it down once the run ends.

This requires the ``dask-cluster`` extra of ``ska-sdp-exec-piper`` (which installs ``ska-sdp-exec-batchlet``) in the pipeline's python environment:

.. code-block:: bash

   pip install --extra-index-url https://artefact.skao.int/repository/pypi-internal/simple \
       "ska-sdp-exec-piper[dask-cluster]>=2.3.1"

The size of the cluster can be controlled with:

``--workers-per-node``
   Number of dask workers started on each node.

``--threads-per-worker``
   Number of threads in each dask worker.

Both options are optional. When either of them is not passed, a value is derived from the CPUs available on the node. Each worker gets an equal share of the node's memory as its memory limit. These options are ignored (with a warning) unless ``--create-dask-cluster`` is set.

The temporary files of the dask workers are written to the output directory of the run.

The kind of cluster which gets created depends on where the pipeline runs.

On a local machine
^^^^^^^^^^^^^^^^^^

Outside a SLURM allocation, a dask ``LocalCluster`` is created on the current machine. Here ``--workers-per-node`` is the total number of workers.

.. code-block:: bash

   ska-sdp-instrumental-calibration run \
       --config config.yml \
       --create-dask-cluster \
       --workers-per-node 4 \
       --threads-per-worker 2 \
       /path/to/data.ms

Inside a SLURM allocation
^^^^^^^^^^^^^^^^^^^^^^^^^

When the pipeline runs inside a SLURM job (that is, when the ``SLURM_JOB_ID`` environment variable is set), a SLURM-based dask cluster is created instead. The dask scheduler runs on the node where the pipeline is started, and the workers are launched with ``srun`` on every node of the allocation. The pipeline waits until all workers have joined before starting the run.

The total number of workers is ``--workers-per-node`` times the number of nodes in the allocation.

A minimal batch script looks like:

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=inst
   #SBATCH --nodes=2
   #SBATCH --time=02:00:00

   # Activate the python environment where the pipeline is installed
   source /path/to/venv/bin/activate

   ska-sdp-instrumental-calibration run \
       --config config.yml \
       --output /path/to/output \
       --create-dask-cluster \
       --workers-per-node 4 \
       --threads-per-worker 8 \
       /path/to/data.ms

The cluster creation can be tuned through these environment variables:

``BATCHLET_DASK_CLUSTER__FORCE_LOCAL``
   If set, a local cluster is created even inside a SLURM allocation.

``BATCHLET_DASK_CLUSTER__DASHBOARD_ADDRESS``
   Address on which the dask dashboard of the created cluster is served, for example ``:8787``.

Connect to an existing dask cluster
-----------------------------------

The pipeline can also submit its work to a dask cluster which you start and manage yourself. Pass the address of its scheduler using ``--dask-scheduler``.

For example, to start a cluster on your local machine with the dask CLI:

.. note::

   Run the scheduler and the workers in the same python environment where the pipeline is installed.

1. Start a dask scheduler. By default, it listens on port ``8786``.

   .. code-block:: bash

      dask scheduler

2. In another terminal, start the workers.

   .. code-block:: bash

      dask worker localhost:8786 --nworkers <NUM_WORKERS> --nthreads <NUM_THREADS>

   For finer control over individual workers, you can instead start each worker in a separate terminal.

3. Run the pipeline, pointing it to the scheduler.

   .. code-block:: bash

      ska-sdp-instrumental-calibration run \
          --config config.yml \
          --dask-scheduler localhost:8786 \
          /path/to/data.ms

Instead of steps 1 and 2, you can run `scripts/dev/ipy_cluster.py <https://gitlab.com/ska-telescope/sdp/science-pipeline-workflows/ska-sdp-instrumental-calibration/-/blob/main/scripts/dev/ipy_cluster.py>`_. It starts a ``LocalCluster``, prints the scheduler and dashboard addresses, and keeps the cluster running (in an IPython shell, if IPython is installed) until you stop it. The cluster size, the ports, and an optional dask performance report are set by the variables at the top of the script.

.. note::

   If both ``--dask-scheduler`` and ``--create-dask-cluster`` are passed, the given scheduler is used, and no cluster is created.

Benchmarking with the dask performance report
---------------------------------------------

The pipeline can capture a `dask performance report <https://docs.dask.org/en/stable/diagnostics-distributed.html#performance-reports>`_ of the run. Pass ``--with-report`` to save the report as ``dask_performance_report.html`` in the output directory of the run.

The report requires:

* a dask cluster, either one created with ``--create-dask-cluster`` or an existing one passed with ``--dask-scheduler``. With local threads, no report is generated.
* the ``dask-report`` extra of ``ska-sdp-exec-piper``, which installs ``bokeh``:

  .. code-block:: bash

     pip install --extra-index-url https://artefact.skao.int/repository/pypi-internal/simple \
         "ska-sdp-exec-piper[dask-report]>=2.3.1"

If either requirement is not met, the pipeline logs a warning and runs without capturing the report.

.. code-block:: bash

   ska-sdp-instrumental-calibration run \
       --config config.yml \
       --output output/benchmark \
       --create-dask-cluster \
       --with-report \
       /path/to/data.ms

.. note::

   The performance report collects detailed metrics for every task, which adds some overhead to the run. Keep this in mind when comparing run times.

Using the helper scripts
------------------------

The ``scripts`` directory of the repository has helper scripts for running the pipeline. Scripts under ``scripts/user`` are meant for anyone running the pipeline. Scripts under ``scripts/dev`` are meant for developers and testers working on the SKA DP HPC cluster.

.. raw:: html

   <details>
   <summary>Scripts in the repository</summary>

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Script
     - Use it to
   * - ``scripts/user/inst.sh``
     - Run any of the INST pipelines locally or as a SLURM job, with log capture, output directory handling, a dask cluster, and optional monitoring. It is self-contained, so it can be downloaded and used on its own. See `Using the inst.sh script`_.
   * - ``scripts/dev/run.sh``
     - Run a git checkout of the pipeline, with its dependencies loaded from the metamodules. It submits ``scripts/user/inst.sh``. See `Using the development code`_.
   * - ``scripts/dev/setup_testdir.sh``
     - Set up a test directory with a clone of the repository, and install extra python packages on top of the metamodules. See `Setting up a test directory`_.
   * - ``scripts/dev/ipy_cluster.py``
     - Start a dask ``LocalCluster`` and keep it running, to be used with ``--dask-scheduler``. See `Connect to an existing dask cluster`_.
   * - ``scripts/dev/_utils.sh``
     - Shared bash functions (logging, module and spack loading, and so on) used by the other dev scripts. Not meant to be run directly.
   * - ``scripts/bin/*``
     - Run the pipeline CLIs from the checkout, without installing them. ``scripts/dev/run.sh`` adds this directory to the ``PATH``.

.. raw:: html

   </details>

Using the inst.sh script
^^^^^^^^^^^^^^^^^^^^^^^^

The `scripts/user/inst.sh <https://gitlab.com/ska-telescope/sdp/science-pipeline-workflows/ska-sdp-instrumental-calibration/-/blob/main/scripts/user/inst.sh>`_ script runs the pipeline through `batchlet <https://developer.skao.int/projects/ska-sdp-exec-batchlet/en/latest/>`_. Compared to calling the pipeline directly, it:

* captures the logs of the script and the pipeline in ``<output-dir>/captured.log``, and mirrors them to the SLURM output file,
* picks a new output directory if the given one already exists (unless ``--reuse-dirs`` is passed),
* starts a dask cluster with ``batchlet``, on all the nodes of the SLURM allocation, and passes it to the pipeline with ``--dask-scheduler``. Pass ``--disable-dask-cluster`` to skip this,
* with ``--enable-monitor``, monitors the resources and logs, and writes the reports to ``<output-dir>/reports``.

The script can run anywhere, as long as the pipeline and ``batchlet`` are in the ``PATH``. Download it with:

.. code-block:: bash

   curl -o inst.sh 'https://gitlab.com/ska-telescope/sdp/science-pipeline-workflows/ska-sdp-instrumental-calibration/-/raw/main/scripts/user/inst.sh'
   chmod +x inst.sh

Run ``./inst.sh --help`` for all its options. The measurement sets are passed as positional arguments. To submit a run on the SKA DP HPC cluster:

.. code-block:: bash

   module purge
   module load ska-sdp-spack py-ska-sdp-exec-batchlet py-ska-sdp-instrumental-calibration

   sbatch --nodes 2 ./inst.sh \
       --config /path/to/config.yml \
       --output-dir /path/to/output \
       --enable-monitor \
       /path/to/data.ms

To run it locally, use ``bash ./inst.sh`` instead of ``sbatch``. To run the target calibration or ionospheric pipelines, pass their command with ``--cmd``.

Using the development code
^^^^^^^^^^^^^^^^^^^^^^^^^^

To test a branch of this repository on the SKA DP HPC cluster, the `scripts/dev/run.sh <https://gitlab.com/ska-telescope/sdp/science-pipeline-workflows/ska-sdp-instrumental-calibration/-/blob/main/scripts/dev/run.sh>`_ script loads the dependencies from the metamodules, and the pipeline from a git checkout.

.. note::

   Do not ``pip install`` the checkout on top of the metamodules. The module loaded packages do not mix well with pip site-packages. ``run.sh`` sets ``PATH`` and ``PYTHONPATH`` to the checkout instead.

1. Clone the repository, and switch to the branch to test:

   .. code-block:: bash

      git clone https://gitlab.com/ska-telescope/sdp/science-pipeline-workflows/ska-sdp-instrumental-calibration.git
      cd ska-sdp-instrumental-calibration
      git checkout <branch-name>

2. Edit the configuration section at the top of ``scripts/dev/run.sh`` (input measurement sets, config path, sky models, SLURM partition, number of nodes, and so on).

3. Run it with bash. Never submit ``run.sh`` itself with ``sbatch``:

   .. code-block:: bash

      bash scripts/dev/run.sh

The script prints the full command and asks for confirmation. It then submits ``scripts/user/inst.sh`` with ``sbatch``. The outputs are written to a directory named after ``SCENARIO``, in the current directory.

``run.sh`` can also be changed to:

* load the dependencies from your own spack installation, by replacing the ``load_env_modules`` call with ``load_spack_modules``,
* run without SLURM, by setting ``ENTRYEXEC=('bash')``.

Setting up a test directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To also test packages which are not in the metamodules (for example, a branch of a dependency), use `scripts/dev/setup_testdir.sh <https://gitlab.com/ska-telescope/sdp/science-pipeline-workflows/ska-sdp-instrumental-calibration/-/blob/main/scripts/dev/setup_testdir.sh>`_. It clones this repository into ``./code/inst``, and installs the listed packages with ``uv pip`` into ``./.pip_packages``, using the python of the metamodules. The packages are installed without their dependencies, so they do not replace the packages of the metamodules.

1. Copy ``setup_testdir.sh`` into an empty test directory.
2. Edit ``REPO_REV`` (the branch or commit to clone) and ``UV_PIP_PACKAGES`` at the top of the script.
3. Run it with bash:

   .. code-block:: bash

      bash setup_testdir.sh

At the end, the script prints the lines to add to the ``ENV_SETUP_SCRIPT`` section of ``scripts/dev/run.sh``, so that the installed packages are picked up. The script also links ``./scripts`` to the scripts of the clone, so ``bash scripts/dev/run.sh`` can be run from the test directory as described above.
