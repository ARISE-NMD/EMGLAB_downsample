from pathlib import Path
import argparse
import time
import json
import dataclasses
from dataclasses import dataclass, field

from helpers import setup_logger, is_step_done
from preprocessor import Preprocessor
from downsampler import Downsampler
from feature_extractor import FeatureExtractor
from feature_selector import FeatureSelector
from classifier import Classifier

@dataclass
class PipelineConfig:
    dataset_path: Path
    save_path: Path
    fc_parameters: str
    cv_folds: int
    boruta_max_iter: int
    force: bool
    downsamplers: list = field(default_factory=lambda: [
        "Original", "Decimate", "LTTB", "MinMax", "M4", "MinMaxLTTB"
    ])
    downsampling_factors: list = field(default_factory=lambda: [
        2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 
        70, 75, 80, 85, 90, 95, 100, 200, 300, 400, 500, 1000
    ])

class DownsamplePipeline:
    def __init__(self, args):
        dataset_path = Path(args.data_dir).resolve()
        save_path = Path(args.output_dir).resolve()

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

        save_path.mkdir(parents=True, exist_ok=True)
        
        self.config = PipelineConfig(
            dataset_path=dataset_path,
            save_path=save_path,
            fc_parameters=args.fc_parameters,
            cv_folds=args.cv_folds,
            boruta_max_iter=args.boruta_max_iter,
            force=args.force
        )

        self.logger = setup_logger(save_path)

        # dump config to json
        with open(save_path / "config.json", "w") as f:
            json.dump(dataclasses.asdict(self.config), f, indent=4, default=str)

        # Define sub-directories for pipeline steps
        self.preprocessed_output = save_path / "preprocessed"
        self.downsampled_output = save_path / "downsampled"
        self.features_output = save_path / "features"
        self.selected_output = save_path / "selected"
        self.classification_output = save_path / "classification"

        # Initialize modules
        self.preprocessor = Preprocessor(self.config, self.logger)
        self.downsampler = Downsampler(self.config, self.logger)
        self.extractor = FeatureExtractor(self.config, self.logger)
        self.selector = FeatureSelector(self.config, self.logger)
        self.classifier = Classifier(self.config, self.logger)

        self.logger.info("Pipeline initialized.")
        self.logger.info(f"Data Source: {dataset_path}")
        self.logger.info(f"Output Directory: {save_path}")

    def run_step(self, name: str, func, *args):
        out_path = args[-1] 
        if self.config.force or not is_step_done(out_path, name):
            out_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"=== Starting step: {name} ===")
            start = time.time()
            func(*args)
            duration = time.time() - start
            self.logger.info(f"=== Finished step: {name} in {duration:.2f}s ===\n")
        else:
            self.logger.info(f"=== Step {name} already done, skipping ===\n")

    def run(self):
        self.run_step("preprocessing", self.preprocessor.run, self.preprocessed_output)
        self.run_step("downsampling", self.downsampler.run, self.preprocessed_output, self.downsampled_output)
        self.run_step("feature_extraction", self.extractor.run, self.downsampled_output, self.features_output)
        self.run_step("feature_selection", self.selector.run, self.preprocessed_output, self.features_output, self.selected_output)
        self.run_step("classification", self.classifier.run, self.selected_output, self.classification_output)

def parse_args():
    parser = argparse.ArgumentParser(description="EMGLAB Downsampling Effect Pipeline")
    
    # Path handling
    parser.add_argument('--data-dir', type=str, required=True, help='Path to the raw dataset folder')
    parser.add_argument('--output-dir', type=str, default='./results', help='Path to save processed results')

    # Logic/Hyperparameters
    parser.add_argument('--fc-parameters', type=str, default='efficient', choices=['minimal', 'efficient', 'comprehensive'])
    parser.add_argument('--cv-folds', type=int, default=10)
    parser.add_argument('--boruta-max-iter', type=int, default=100)
    parser.add_argument('--force', action='store_true', help='Force re-run all steps')

    return parser.parse_args()

def main():
    args = parse_args()
    pipeline = DownsamplePipeline(args)
    pipeline.run()

if __name__ == "__main__":
    main()