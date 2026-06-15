"""
Dataset class for respiratory flow prediction.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class RespiratoryFlowDataset(Dataset):
    """
    Dataset for paired Mel spectrograms and flow curves.
    
    Args:
        mel_dir: Path to Mel spectrogram directory
        csv_dir: Path to flow curve directory
        label_file: Path to pulmonary function label CSV
        sample_ids: List of sample IDs to include
    
    Returns:
        Dictionary containing:
            - mel: Tensor (60, 128) - Mel spectrogram
            - flow: Tensor (60,) - Flow curve
            - labels: Tensor (3,) - [FEV1, FVC, PEF]
            - demographic: Optional[Tensor] - demographic feature vector (when enabled)
            - subject_id: str - Subject identifier
            - sample_id: str - Sample identifier
    """
    
    def __init__(self, mel_dir, csv_dir, label_file, sample_ids=None,
                 demographic_features=None, training=False,
                 spec_augment=False, time_mask_param=5, freq_mask_param=15,
                 num_time_masks=1, num_freq_masks=1):
        """
        Initialize dataset.
        
        Args:
            mel_dir: Directory containing Mel spectrograms (.npy files)
            csv_dir: Directory containing flow curves (.csv files)
            label_file: Path to label CSV file
            sample_ids: List of sample IDs to include (if None, use all)
            demographic_features: List of demographic features to use, e.g., ['gender'], ['age'], etc.
                                 If None, use all features ['gender', 'age', 'height', 'weight']
            training: Whether this dataset is used for training (enables augmentation)
            spec_augment: Whether to apply SpecAugment on Mel spectrograms
            time_mask_param: Maximum width of each time mask (T in SpecAugment)
            freq_mask_param: Maximum width of each frequency mask (F in SpecAugment)
            num_time_masks: Number of time masks to apply
            num_freq_masks: Number of frequency masks to apply
        """
        self.mel_dir = mel_dir
        self.csv_dir = csv_dir
        self.label_file = label_file
        self.training = bool(training)
        self.spec_augment = bool(spec_augment) and self.training
        self.time_mask_param = int(time_mask_param)
        self.freq_mask_param = int(freq_mask_param)
        self.num_time_masks = int(num_time_masks)
        self.num_freq_masks = int(num_freq_masks)
        
        # Set demographic features to use.
        # Passing [] makes the dataset omit the optional demographic tensor.
        if demographic_features is None:
            demographic_features = ['gender', 'age', 'height', 'weight']
        self.demographic_features = list(demographic_features)
        
        # Load labels
        self.labels_df = pd.read_csv(label_file)
        
        # Create label lookup dictionary for faster access
        # Convert ID to zero-padded string for quick lookup
        self.label_dict = {}
        for _, row in self.labels_df.iterrows():
            subject_id = f"{int(row['id']):04d}"
            # Calculate BMI: BMI = weight (kg) / (height (m))^2
            # Height is in cm, so convert to meters: height/100
            height = float(row.get('high', row.get('height', 0)))
            weight = float(row.get('weight', 0))
            if height > 0:
                bmi = weight / ((height / 100.0) ** 2)
            else:
                bmi = 0.0
            
            self.label_dict[subject_id] = {
                'fev1': row.get('FEV1', row.get('fev1')),
                'fvc': row.get('FVC', row.get('fvc')),
                'pef': row.get('PEF', row.get('pef')),
                # Demographic information
                'gender': row.get('gender', 0),  # 0 or 1
                'age': row.get('age', 0),
                'height': height,
                'weight': weight,
                'bmi': bmi
            }
        
        # Get available sample IDs
        if sample_ids is None:
            # Use all samples that have both mel and csv files
            mel_files = set([f.replace('.npy', '') for f in os.listdir(mel_dir) if f.endswith('.npy')])
            csv_files = set([f.replace('.csv', '') for f in os.listdir(csv_dir) if f.endswith('.csv')])
            available_ids = mel_files.intersection(csv_files)
            self.sample_ids = sorted(list(available_ids))
        else:
            self.sample_ids = sample_ids
        
        # Filter out samples not in labels
        # Extract subject IDs from sample IDs and match with label IDs
        # Sample IDs are like "0001_1", "0001_2", etc. (zero-padded strings)
        # Label IDs are integers like 1, 2, 3, etc.
        # Convert label IDs to zero-padded 4-digit strings for matching
        label_ids = set(self.label_dict.keys())
        filtered_ids = []
        for sid in self.sample_ids:
            subject_id = sid.split('_')[0]  # Extract "0001" from "0001_1"
            if subject_id in label_ids:
                filtered_ids.append(sid)
        
        self.sample_ids = filtered_ids
        
        logger.info(f"Dataset initialized with {len(self.sample_ids)} samples")
    
    _MAX_LOAD_RETRIES = 3

    def _apply_spec_augment(self, mel: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment (time + frequency masking) to Mel spectrogram.
        
        Args:
            mel: (T, n_mels) Mel spectrogram tensor
        Returns:
            Augmented Mel spectrogram of the same shape
        """
        mel = mel.clone()
        T, F = mel.shape

        for _ in range(self.num_time_masks):
            t = torch.randint(0, max(self.time_mask_param, 1), (1,)).item()
            t0 = torch.randint(0, max(T - t, 1), (1,)).item()
            mel[t0:t0 + t, :] = 0.0

        for _ in range(self.num_freq_masks):
            f = torch.randint(0, max(self.freq_mask_param, 1), (1,)).item()
            f0 = torch.randint(0, max(F - f, 1), (1,)).item()
            mel[:, f0:f0 + f] = 0.0

        return mel

    def __len__(self):
        """Return number of samples."""
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Args:
            idx: Index of sample
        
        Returns:
            Dictionary with mel, flow, labels, subject_id, sample_id
        """
        for attempt in range(self._MAX_LOAD_RETRIES + 1):
            current_idx = (idx + attempt) % len(self)
            sample_id = self.sample_ids[current_idx]

            try:
                mel_path = os.path.join(self.mel_dir, f"{sample_id}.npy")
                mel = np.load(mel_path)  # Shape: (128, 60)
                mel = mel.T  # Transpose to (60, 128)

                csv_path = os.path.join(self.csv_dir, f"{sample_id}.csv")
                flow_data = np.loadtxt(csv_path, delimiter=',')
                flow = flow_data[:, 1]

                subject_id = sample_id.split('_')[0]
                label_info = self.label_dict[subject_id]
                labels = np.array(
                    [label_info['fev1'], label_info['fvc'], label_info['pef']],
                    dtype=np.float32,
                )

                mel = torch.from_numpy(mel).float()
                if self.spec_augment:
                    mel = self._apply_spec_augment(mel)
                flow = torch.from_numpy(flow).float()
                labels = torch.from_numpy(labels).float()

                sample = {
                    'mel': mel,
                    'flow': flow,
                    'labels': labels,
                    'subject_id': subject_id,
                    'sample_id': sample_id,
                }

                if len(self.demographic_features) > 0:
                    demographic_list = []
                    feature_mapping = {
                        'gender': 'gender',
                        'age': 'age',
                        'height': 'height',
                        'weight': 'weight',
                        'bmi': 'bmi',
                    }
                    for feature in self.demographic_features:
                        if feature in feature_mapping:
                            demographic_list.append(float(label_info[feature_mapping[feature]]))

                    demographic = np.array(demographic_list, dtype=np.float32)
                    sample['demographic'] = torch.from_numpy(demographic).float()

                return sample

            except Exception as e:
                logger.warning(f"Error loading sample {sample_id}: {e}")
                if attempt >= self._MAX_LOAD_RETRIES:
                    raise RuntimeError(
                        f"Failed to load data after {self._MAX_LOAD_RETRIES + 1} "
                        f"consecutive attempts starting from index {idx}"
                    ) from e

