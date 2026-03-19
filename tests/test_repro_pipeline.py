import shutil
import tempfile
import unittest
from pathlib import Path

from meta_controller.analysis.make_figures import main as figures_main
from meta_controller.analysis.make_tables import main as tables_main
from meta_controller.experiments.evaluate import main as eval_main
from meta_controller.experiments.train import main as train_main

from tests.support import write_calibration_csv


class ReproPipelineTests(unittest.TestCase):
    def test_train_eval_and_analysis_pipeline(self) -> None:
        temp_dir = Path(tempfile.mkdtemp())
        try:
            results_dir = temp_dir / "results"
            output_dir = temp_dir / "artifacts"
            calibration_csv = write_calibration_csv(temp_dir / "calibration.csv")
            calibration_report = temp_dir / "calibration_report.json"
            train_main(
                [
                    "--config",
                    "configs/training.yaml",
                    "--results-dir",
                    str(results_dir),
                    "--set",
                    f"env.calibration.csv_path={calibration_csv}",
                    "--set",
                    f"env.calibration.report_path={calibration_report}",
                ]
            )
            eval_main(["--config", "configs/evaluation.yaml", "--results-dir", str(results_dir)])
            figures_main(["--results-dir", str(results_dir), "--output-dir", str(output_dir)])
            tables_main(["--results-dir", str(results_dir), "--output-dir", str(output_dir)])
            self.assertTrue((output_dir / "Fig1.svg").exists())
            self.assertTrue((output_dir / "Fig2.svg").exists())
            self.assertTrue((output_dir / "Table1.csv").exists())
            self.assertTrue(calibration_report.exists())
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
