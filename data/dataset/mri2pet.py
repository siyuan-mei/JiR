from typing import Union, List, Callable, Tuple
import os
from torch.utils.data import Dataset
from data.utils import load_image


class_dict = {'AD': 0, 'CN': 1, 'PMCI': 2, 'SMCI': 3}

class Mri2PetDataset(Dataset):
    def __init__(self,
                 data_root: str,
                 split_path: str = 'train',  # 'train', 'validation', 'test'
                 disease_class: Union[str, Tuple, List] = ('AD', 'CN', 'PMCI', 'SMCI'),
                 img_suffix: str = '.nii.gz',
                 transform: Callable = None):

        self.data_root = data_root
        self.split_path = split_path
        self.img_suffix = img_suffix
        self.transform = transform
        self.disease_class = disease_class
        self.samples = self._gather_samples()

    def _gather_samples(self):
        samples = []
        for cls in self.disease_class:
            cls_dir = os.path.join(self.data_root, self.split_path, cls)
            if not os.path.exists(cls_dir):
                print(f"Class directory not found: {cls_dir}")
                continue
            patient_dirs = sorted(
                [
                    os.path.join(cls_dir, d)
                    for d in os.listdir(cls_dir)
                    if os.path.isdir(os.path.join(cls_dir, d))
                ]
            )
            for p in patient_dirs:
                p_id = os.path.basename(p)
                mr_path = os.path.join(p, p_id + "_M00_T1w" + self.img_suffix)
                pet_path = os.path.join(p, p_id + "_M00_fdg_pet"  + self.img_suffix)
                if not os.path.exists(mr_path) or not os.path.exists(pet_path):
                    raise FileNotFoundError(
                        f"Missing MRI/PET pair for patient '{p_id}': {mr_path}, {pet_path}"
                    )
                samples.append({
                    'input': mr_path,
                    'target': pet_path,
                    'class': cls,
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

        expected_shape = (1, 112, 128, 112)
        if input_img.shape != expected_shape or target_img.shape != expected_shape:
            raise ValueError(
                f"Expected MRI/PET shape {expected_shape}, got "
                f"input={input_img.shape}, target={target_img.shape} for id={sample['id']}."
            )

        data = {
            'input': input_img,
            'target': target_img,
            # 'mr_code': 0,
            # 'pet_code': 1,
            'cls_code': class_dict[cls],
            'class': cls,
            'id': sample['id'],
        }

        if self.transform:
            data = self.transform(data)
        return data

