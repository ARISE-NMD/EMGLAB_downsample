from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.inspection import permutation_importance
from helpers import format_feature_names, mark_done, mark_undone
from pathlib import Path
import pandas as pd
import numpy as np


class Classifier:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def calculate_per_class_metrics(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        classes = np.unique(y_true)
        
        per_class_metrics = {}
        
        for i, class_label in enumerate(classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - (tp + fn + fp)
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            per_class_metrics[f'sensitivity_class_{class_label}'] = sensitivity
            per_class_metrics[f'specificity_class_{class_label}'] = specificity
            
        return per_class_metrics

    def run(self, input_path: Path, output_path: Path):
            mark_undone(output_path, "classification")
            self.logger.info("Running classification...")

            metrics_all = []
            all_feature_importances = []

            for downsampler in self.config.downsamplers:
                if downsampler == 'Original':
                    downsampling_factors = [1]
                else:
                    downsampling_factors = self.config.downsampling_factors
                for factor in downsampling_factors:
                    self.logger.info(f"Processing: {downsampler} (factor={factor})")
        
                    current_dir = output_path / f"{downsampler}/{factor}"
                    current_dir.mkdir(parents=True, exist_ok=True)

                    fold_preds, fold_probs, fold_truths = [], [], []
                    current_factor_metrics = [] 

                    for fold in range(self.config.cv_folds):
                        input_fold_path = input_path / f"{downsampler}/{factor}/fold_{fold+1}"

                        X_train = pd.read_csv(input_fold_path / "X_train.csv")
                        X_test = pd.read_csv(input_fold_path / "X_test.csv")
                        y_train = np.load(input_fold_path / "y_train.npy")
                        y_test = np.load(input_fold_path / "y_test.npy")

                        # Train classifier
                        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
                        clf.fit(X_train, y_train)

                        # Predict
                        y_pred = clf.predict(X_test)
                        y_proba = clf.predict_proba(X_test)

                        # Calculate per-class metrics
                        per_class_metrics = self.calculate_per_class_metrics(y_test, y_pred)

                        # Store metrics
                        metrics = {
                            "downsampler": downsampler,
                            "factor": factor,
                            "fold": fold + 1,
                            "accuracy": accuracy_score(y_test, y_pred),
                            "f1": f1_score(y_test, y_pred, average="macro"),
                            "precision": precision_score(y_test, y_pred, average="macro"),
                            "recall": recall_score(y_test, y_pred, average="macro"),
                            "roc_auc_ovr": roc_auc_score(y_test, y_proba, multi_class="ovr"),
                            "roc_auc_ovo": roc_auc_score(y_test, y_proba, multi_class="ovo"),
                            **per_class_metrics
                        }
                        
                        metrics_all.append(metrics)
                        current_factor_metrics.append(metrics)

                        fold_preds.append(y_pred)
                        fold_probs.append(y_proba)
                        fold_truths.append(y_test)

                        impurity_result = clf.feature_importances_
                        impurity_importance = pd.Series(impurity_result, index=X_train.columns)
                        
                        perm_result = permutation_importance(clf, X_test, y_test, scoring="f1_macro", n_jobs=-1)
                        perm_importance = pd.Series(perm_result.importances_mean, index=X_test.columns)

                        importance_data = pd.DataFrame({
                            'feature_name': format_feature_names(impurity_importance.index),
                            'importance_impurity': impurity_importance.values,
                            'importance_permutation': perm_importance.values,
                            'downsampler': downsampler,
                            'factor': factor,
                            'fold': fold + 1
                        })
                        all_feature_importances.append(importance_data)

                    accs = [m['accuracy'] for m in current_factor_metrics]
                    self.logger.info(f"Metrics: acc:{np.mean(accs):.4f} ± {np.std(accs):.4f}")

                    np.save(current_dir / "y_true.npy", np.concatenate(fold_truths))
                    np.save(current_dir / "y_pred.npy", np.concatenate(fold_preds))
                    np.save(current_dir / "y_probs.npy", np.concatenate(fold_probs))

            # Save all raw metrics (no aggregation)
            metrics_df = pd.DataFrame(metrics_all)
            metrics_df.to_csv(output_path / "metrics_raw.csv", index=False)
            
            # Save all raw feature importances
            all_importances_df = pd.concat(all_feature_importances, ignore_index=True)
            all_importances_df.to_csv(output_path / "feature_importances_raw.csv", index=False)
            
            mark_done(output_path, "classification")
