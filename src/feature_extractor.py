from pathlib import Path
from helpers import mark_done, mark_undone
import numpy as np
import pandas as pd
from tsfresh import extract_features as ts_extract_features
from tsfresh.feature_extraction.settings import MinimalFCParameters, EfficientFCParameters, ComprehensiveFCParameters


class FeatureExtractor:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.mapping = {
            'efficient': EfficientFCParameters,
            'minimal': MinimalFCParameters,
            'comprehensive': ComprehensiveFCParameters
        }

    def run(self, input_path: Path, output_path: Path):
        mark_undone(output_path, "feature_extraction")
        self.logger.info(f"Running feature extraction with fc_parameter={self.config.fc_parameters}...")

        metrics = []
        for downsampler in self.config.downsamplers:
            current_dir = output_path / downsampler
            current_dir.mkdir(parents=True, exist_ok=True)
            if downsampler == 'Original':
                downsampling_factors = [1]
            else:
                downsampling_factors = self.config.downsampling_factors
            for factor in downsampling_factors:
                # Get into tsfresh shape
                X = np.load(input_path / f"{downsampler}/X_{factor}.npy")
                n_samples, n_timesteps = X.shape
                df = pd.DataFrame({
                    'id': np.repeat(np.arange(n_samples), n_timesteps),
                    'time': np.tile(np.arange(n_timesteps), n_samples),
                    'value': X.flatten()
                })

                profiling_filename = output_path / f"{downsampler}/profiling_{factor}.txt"

                # extract features
                X_feat_df = ts_extract_features(df,
                                                default_fc_parameters=self.mapping[self.config.fc_parameters](),
                                                column_id='id',
                                                column_sort='time',
                                                column_value='value',
                                                show_warnings=False,
                                                n_jobs=3,
                                                profiling_filename=profiling_filename,
                                                profile=True if profiling_filename else False,
                                                disable_progressbar=True)

                self.logger.info(f"{downsampler}({factor}), X.shape: {X.shape} -> {X_feat_df.shape}")
                X_feat_df.to_csv(output_path / f"{downsampler}/X_{factor}_features.csv", index=False)
                
                # load the profiling data and add to the metrics
                if profiling_filename.exists():
                    with open(profiling_filename) as profiling_file:
                        exec_time = float(profiling_file.readlines()[0].split(' ')[-2])
                        metrics.append({
                            'downsampler': downsampler,
                            'factor': factor,
                            'exec_time': exec_time
                        })

        # save metrics
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(output_path / "metrics.csv", index=False)
        mark_done(output_path, "feature_extraction")
