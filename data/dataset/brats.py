from typing import Callable
import os
from torch.utils.data import Dataset
from data.utils import load_image

class_dict = {'t1': 0, 't1ce': 1}

class BratsDataset(Dataset):
    def __init__(self,
                 data_root: str,
                 split_path: str = 'train',  # 'train', 'validation', 'test'
                 img_suffix: str = '.nii.gz',
                 transform: Callable = None):

        self.data_root = data_root
        self.split_path = split_path
        self.img_suffix = img_suffix
        self.transform = transform
        self.samples = self._gather_samples()

    def _gather_samples(self):
        samples = []
        split_dir = os.path.join(self.data_root, self.split_path)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        patient_dirs = sorted(
            [
                os.path.join(split_dir, d)
                for d in os.listdir(split_dir)
                if os.path.isdir(os.path.join(split_dir, d))
            ]
        )
        for p in patient_dirs:
            p_id = os.path.basename(p)
            input_path = os.path.join(p, p_id + "_t1" + self.img_suffix)
            target_path = os.path.join(p, p_id + "_t1ce"  + self.img_suffix)
            if not os.path.exists(input_path) or not os.path.exists(target_path):
                raise FileNotFoundError(
                    f"Missing BraTS pair for patient '{p_id}': {input_path}, {target_path}"
                )
            samples.append({
                'input': input_path,
                'target': target_path,
                'class': "t1",
                'id': p_id,
            })
        if not samples:
            raise RuntimeError(
                f"No samples found under data_root={self.data_root}, split={self.split_path}."
            )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        cls = sample['class']
        input_img = load_image(sample['input'])[None, ...]
        target_img = load_image(sample['target'])[None, ...]

        data = {
            'input': input_img,
            'target': target_img,
            'cls_code': class_dict[cls],
            'class': cls,
            'id': sample['id'],
        }

        if self.transform:
            data = self.transform(data)
        return data

