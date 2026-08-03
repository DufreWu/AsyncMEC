import csv
import os


class ExperimentLogger:

    def __init__(self, path):
        # ----------------------------------
        # Create directory automatically
        # ----------------------------------
        os.makedirs(
            os.path.dirname(path),
            exist_ok=True
        )

        self.file = open(
            path,
            "w",
            newline=""
        )

        self.writer = csv.writer(self.file)

        # ----------------------------------
        # CSV Header
        # ----------------------------------
        self.writer.writerow([

            # Workload
            "time",

            # Actions
            "cpu_freq",
            "gpu_freq",
            "speed",

            # Workload
            "complexity",
            "target_fps",
            "fps",

            # Power
            "p_comp",
            "p_mech",
            "p_total",
            "eng_dist",

            # Mission
            "distance",

            # Battery
            "voltage",
            "current",
            "temperature",
            "soc",
            "soh",
            "capacity"
        ])

        self.file.flush()

    def log(self, row):

        self.writer.writerow(row)
        self.file.flush()

    def close(self):

        if not self.file.closed:
            self.file.close()