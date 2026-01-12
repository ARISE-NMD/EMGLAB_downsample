import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from boruta import BorutaPy
from pathlib import Path

from helpers import mark_done, mark_undone

class FeatureSelector:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def run(self, preprocess_path: Path, input_path: Path, output_path: Path):
        mark_undone(output_path, "feature_selection")
        self.logger.info(f"Running feature selection with CV (folds={self.config.cv_folds}), boruta_max_iter={self.config.boruta_max_iter}...")

        all_metrics = None

        for downsampler in self.config.downsamplers:
            if downsampler == 'Original':
                downsampling_factors = [1]
            else:
                downsampling_factors = self.config.downsampling_factors
            for factor in downsampling_factors:
                X = pd.read_csv(input_path / f"{downsampler}/X_{factor}_features.csv")
                y = np.load(preprocess_path / "y.npy")

                X.fillna(0, inplace=True)

                for i, (train_idx, test_idx) in enumerate(StratifiedKFold(n_splits=self.config.cv_folds).split(X, y)):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    # Run Boruta on training set
                    rf = RandomForestClassifier(n_jobs=-1, n_estimators=1000)
                    boruta = BorutaPy(rf, n_estimators='auto', alpha=0.05, max_iter=self.config.boruta_max_iter, verbose=0)

                    boruta.fit(X_train.values, y_train)
                    selected = boruta.support_
                    selected_names = X.columns[selected]
                    self.logger.info(f"{downsampler}({factor}), fold={i+1}, selected features: {X.shape[1]} -> {len(selected_names)}")

                    # Save reduced feature sets
                    fold_path = output_path / f"{downsampler}/{factor}/fold_{i+1}"
                    fold_path.mkdir(parents=True, exist_ok=True)
                    X_train[selected_names].to_csv(fold_path / "X_train.csv", index=False)
                    X_test[selected_names].to_csv(fold_path / "X_test.csv", index=False)
                    np.save(fold_path / "y_train.npy", y_train)
                    np.save(fold_path / "y_test.npy", y_test)

                    metrics = pd.DataFrame({
                        'downsampler': downsampler,
                        'downsample_factor': factor,
                        'fold': i + 1,
                        'before': X.shape[1],
                        'after': len(selected_names),
                    }, index=[0])

                    if all_metrics is None:
                        all_metrics = metrics
                    else:
                        all_metrics = pd.concat((all_metrics, metrics), axis=0)

        # aggregate metrics across all folds
        all_metrics = all_metrics.groupby(['downsampler', 'downsample_factor']).agg({'before': 'mean', 'after': 'mean'}).reset_index()
        all_metrics['feature_reduction'] = all_metrics['before'] - all_metrics['after']
        all_metrics['feature_reduction_norm'] = (all_metrics['before'] - all_metrics['after']) / all_metrics['before']

        all_metrics.to_csv(output_path / "metrics.csv", index=False)
        mark_done(output_path, "feature_selection")
