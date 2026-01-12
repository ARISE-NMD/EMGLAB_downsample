import os
import json
import numpy as np
import wfdb
from pathlib import Path
from sklearn.utils import shuffle
from helpers import mark_done, mark_undone

class Preprocessor:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.class_mapping = {0: "ALS", 1: "Control", 2: "Myopathic"}
        self.class_assignment = {'A': 0, 'C': 1, 'M': 2}

    def _load_emglab_data(self, dataset_location):
        X, Y = [], []
        fs = None

        files = sorted([f for f in os.listdir(dataset_location) if f.endswith('.bin')])
        
        if not files:
            raise FileNotFoundError(f"No .bin files found in {dataset_location}")

        for file_name in files:
            file_path = os.path.join(dataset_location, file_name)
            record = wfdb.rdrecord(file_path[:-4])

            if fs is None:
                fs = record.fs
            elif fs != record.fs:
                self.logger.warning(f"Inconsistent sampling rate in {file_name}")

            signal = np.nan_to_num(record.p_signal, nan=0.0)
            X.append(signal)
            
            label_char = file_name[5].upper()
            if label_char in self.class_assignment:
                Y.append(self.class_assignment[label_char])
            else:
                self.logger.error(f"Unknown class character '{label_char}' in {file_name}")

        return X, np.array(Y), fs

    def _split_into_fragments(self, X, y, fs, fragment_duration=2):
        samples_per_fragment = int(fragment_duration * fs)
        fragments = []
        labels = []

        for signal, label in zip(X, y):
            num_fragments = len(signal) // samples_per_fragment
            for i in range(num_fragments):
                start = i * samples_per_fragment
                end = start + samples_per_fragment
                fragment = signal[start:end]
                
                if not np.all(fragment == 0):
                    fragments.append(fragment)
                    labels.append(label)

        return np.array(fragments).squeeze(), np.array(labels)

    def run(self, output_path: Path):
        mark_undone(output_path, "preprocessing")
        dataset_path = self.config.dataset_path

        self.logger.info(f"Loading raw EMGLAB data from: {dataset_path}")
        X_raw, Y_raw, fs = self._load_emglab_data(dataset_path)

        self.logger.info(f"Splitting into {2}-second fragments at {fs}Hz...")
        X_frag, y_frag = self._split_into_fragments(X_raw, Y_raw, fs)

        self.logger.info("Shuffling fragments...")
        X_frag, y_frag = shuffle(X_frag, y_frag)

        output_path.mkdir(parents=True, exist_ok=True)

        self.logger.info("Saving processed data...")
        np.save(output_path / "X.npy", X_frag)
        np.save(output_path / "y.npy", y_frag)

        unique, counts = np.unique(y_frag, return_counts=True)
        stats = {self.class_mapping[u]: int(c) for u, c in zip(unique, counts)}

        metadata = {
            "fs": fs,
            "class_mapping": self.class_mapping,
            "num_samples_per_class": stats,
            "samples_per_fragment": X_frag.shape[1],
            "total_fragments": len(X_frag),
        }

        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        self.logger.info(f"Preprocessing complete. {len(X_frag)} fragments saved.")
        mark_done(output_path, "preprocessing")