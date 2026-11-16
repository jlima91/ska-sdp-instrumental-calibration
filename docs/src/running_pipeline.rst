Running the Pipeline
====================

The INST pipelines use `dask <https://docs.dask.org>`_ for their
computations. Depending on the options passed to the ``run`` subcommand,
the computation can run in one of three ways:

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
     - Distributed runs on a single machine, or across the nodes of a
       SLURM allocation, without managing a cluster yourself.
   * - Existing dask cluster
     - ``--dask-scheduler <address>``
     - When you want full control over the scheduler and workers, or want
       to reuse a long-running cluster.

The examples below assume that the pipeline is installed and that a
configuration file has been generated using the ``install-config``
subcommand (see :doc:`Getting Started <README>`).

The examples use ``ska-sdp-instrumental-calibration``. The ``run``
subcommands of ``ska-sdp-instrumental-target-calibration``,
``ska-sdp-instrumental-target-ionospheric``, and the ``experimental``
subcommand accept the same options.

Run with local threads
----------------------

When neither ``--dask-scheduler`` nor ``--create-dask-cluster`` is passed,
no dask cluster is used. Dask computes the task graph locally, in the
pipeline process, using a thread pool.

.. code-block:: bash

   ska-sdp-instrumental-calibration run \
       --config config.yml \
       /path/to/data.ms

Let the pipeline create a dask cluster
--------------------------------------

With ``--create-dask-cluster``, the pipeline starts its own dask cluster
before the run, and shuts it down once the run ends.

This requires the ``dask-cluster`` extra of ``ska-sdp-exec-piper``
(which installs ``ska-sdp-exec-batchlet``) in the pipeline's python
environment:

.. code-block:: bash

   pip install --extra-index-url https://artefact.skao.int/repository/pypi-internal/simple \
       "ska-sdp-exec-piper[dask-cluster]>=2.3.1"

The size of the cluster can be controlled with:

``--workers-per-node``
   Number of dask workers started on each node.

``--threads-per-worker``
   Number of threads in each dask worker.

Both options are optional. When either of them is not passed, a value is
derived from the CPUs available on the node. Each worker gets an equal
share of the node's memory as its memory limit. These options are ignored
(with a warning) unless ``--create-dask-cluster`` is set.

The temporary files of the dask workers are written to the output
directory of the run.

The kind of cluster which gets created depends on where the pipeline runs.

On a local machine
^^^^^^^^^^^^^^^^^^

Outside a SLURM allocation, a dask ``LocalCluster`` is created on the
current machine. Here ``--workers-per-node`` is the total number of
workers.

.. code-block:: bash

   ska-sdp-instrumental-calibration run \
       --config config.yml \
       --create-dask-cluster \
       --workers-per-node 4 \
       --threads-per-worker 2 \
       /path/to/data.ms

Inside a SLURM allocation
^^^^^^^^^^^^^^^^^^^^^^^^^

When the pipeline runs inside a SLURM job (that is, when the
``SLURM_JOB_ID`` environment variable is set), a SLURM-based dask cluster
is created instead. The dask scheduler runs on the node where the pipeline
is started, and the workers are launched with ``srun`` on every node of the
allocation. The pipeline waits until all workers have joined before
starting the run.

The total number of workers is ``--workers-per-node`` times the number of
nodes in the allocation.

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
   Address on which the dask dashboard of the created cluster is served,
   for example ``:8787``.

Connect to an existing dask cluster
-----------------------------------

The pipeline can also submit its work to a dask cluster which you start
and manage yourself. Pass the address of its scheduler using
``--dask-scheduler``.

For example, to start a cluster on your local machine with the dask CLI:

.. note::

   Run the scheduler and the workers in the same python environment where
   the pipeline is installed.

1. Start a dask scheduler. By default, it listens on port ``8786``.

   .. code-block:: bash

      dask scheduler

2. In another terminal, start the workers.

   .. code-block:: bash

      dask worker localhost:8786 --nworkers <NUM_WORKERS> --nthreads <NUM_THREADS>

   For finer control over individual workers, you can instead start each
   worker in a separate terminal.

3. Run the pipeline, pointing it to the scheduler.

   .. code-block:: bash

      ska-sdp-instrumental-calibration run \
          --config config.yml \
          --dask-scheduler localhost:8786 \
          /path/to/data.ms

.. note::

   If both ``--dask-scheduler`` and ``--create-dask-cluster`` are passed,
   the given scheduler is used, and no cluster is created.

Benchmarking with the dask performance report
---------------------------------------------

The pipeline can capture a `dask performance report
<https://docs.dask.org/en/stable/diagnostics-distributed.html#performance-reports>`_
of the run. Pass ``--with-report`` to save the report as
``dask_performance_report.html`` in the output directory of the run.

The report requires:

* a dask cluster, either one created with ``--create-dask-cluster`` or an
  existing one passed with ``--dask-scheduler``. With local threads, no
  report is generated.
* the ``dask-report`` extra of ``ska-sdp-exec-piper``, which installs
  ``bokeh``:

  .. code-block:: bash

     pip install --extra-index-url https://artefact.skao.int/repository/pypi-internal/simple \
         "ska-sdp-exec-piper[dask-report]>=2.3.1"

If either requirement is not met, the pipeline logs a warning and runs
without capturing the report.

.. code-block:: bash

   ska-sdp-instrumental-calibration run \
       --config config.yml \
       --output output/benchmark \
       --create-dask-cluster \
       --with-report \
       /path/to/data.ms

.. note::

   The performance report collects detailed metrics for every task, which
   adds some overhead to the run. Keep this in mind when comparing run
   times.
