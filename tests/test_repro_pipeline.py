import shutil
import tempfile
import unittest
from pathlib import Path

from meta_controller.analysis.make_figures import main as figures_main
from meta_controller.analysis.make_tables import main as tables_main
from meta_controller.experiments.evaluate import main as eval_main
from meta_controller.experiments.train import main as train_main


class ReproPipelineTests(unittest.TestCase):
    def test_train_eval_and_analysis_pipeline(self) -> None:
        temp_dir = Path(tempfile.mkdtemp())
        try:
            results_dir = temp_dir / 'results'
            output_dir = temp_dir / 'artifacts'
            train_main(['--config', 'configs/training.yaml', '--results-dir', str(results_dir)])
            eval_main(['--config', 'configs/evaluation.yaml', '--results-dir', str(results_dir)])
            figures_main(['--results-dir', str(results_dir), '--output-dir', str(output_dir)])
            tables_main(['--results-dir', str(results_dir), '--output-dir', str(output_dir)])
            self.assertTrue((output_dir / 'Fig1.svg').exists())
            self.assertTrue((output_dir / 'Fig2.svg').exists())
            self.assertTrue((output_dir / 'Table1.csv').exists())
        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
