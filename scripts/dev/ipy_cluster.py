#!/usr/bin/env python3

"""
A python script to create a LocalCluster, and wait until user interrupts.

If IPython is installed, it will start a ipython
shell (to block and to inspect) in the current terminal.
You can turn this behavior off by changing IPYTHON_EMBED toggle.

This can also generate dask performance reports, if enabled via the
ENABLE_PERF_REPORT toggle.

Usage
-----
# Basic usage
>>> python3 ipy_cluster.py
# To redirect all worker logs to a file
>>> python3 ipy_cluster.py &> cluster.log
"""

import os
import time
from contextlib import ExitStack

from distributed import LocalCluster, performance_report

try:
    from IPython import embed

    IPYTHON_EMBED = True
except ImportError:
    IPYTHON_EMBED = False

threads_per_worker = 4
dashboard_address = ":30088"
scheduler_port = 34567
worker_scratch_directory = "./.temp/"
resources_per_worker = {"process": threads_per_worker}

ENABLE_PERF_REPORT = False
report_file_path = "cluster-dask-report.html"


if __name__ == "__main__":
    with ExitStack() as stack:
        cluster: LocalCluster = stack.enter_context(
            LocalCluster(
                threads_per_worker=threads_per_worker,
                dashboard_address=dashboard_address,
                scheduler_port=scheduler_port,
                local_directory=worker_scratch_directory,
                resources=resources_per_worker,
            )
        )
        client = stack.enter_context(cluster.get_client())

        print("cluster.worker_spec: ", cluster.new_spec)
        print("cluster.n_workers: ", len(cluster.workers))
        print("cluster.scheduler_address: ", cluster.scheduler_address)
        print("cluster.dashboard_link: ", cluster.dashboard_link)

        if ENABLE_PERF_REPORT:
            if os.path.exists(report_file_path):
                os.remove(report_file_path)

            stack.enter_context(performance_report(filename=report_file_path))

        if IPYTHON_EMBED:
            print(
                "\nEntering IPython shell... Run '%who' to see list of defined vars\n"
            )
            embed(colors="Linux")
        else:
            try:
                print("\nCluster started, interrupt to close...\n")
                while True:
                    time.sleep(3600)
            except KeyboardInterrupt:
                print("\nProgram stopped gracefully via KeyboardInterrupt.")
