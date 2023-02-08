from torch.utils.data import Dataset
import nibabel as nib
from os.path import join
import numpy as np
import pandas as pd


class niiDataset(Dataset):

  def __init__(self, annotations_file, path_to_dir, min_card_age, 
               training=False, validation=False, transform=None, target_transform=None):
    
    assert min_card_age == 40 or min_card_age == 50

    self.labels_csv = pd.read_csv(join(path_to_dir, annotations_file), sep=',')

    if training:
      self.img_labels = self.labels_csv[self.labels_csv[f'train_{min_card_age}'] == 1]
    elif validation:
      self.img_labels = self.labels_csv[self.labels_csv[f'val_{min_card_age}'] == 1]

    self.path_to_dir = path_to_dir
    self.img_dir = join(self.path_to_dir, 'nii')

    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    return len(self.img_labels)

  def __getitem__(self, idx):

    img_path = join(self.img_dir, self.img_labels.iloc[idx]["file_name"])
    image = np.squeeze(np.asanyarray(nib.load(img_path).dataobj))
    image_cropped = image[2:66, 3:67, 3:67]

    label = self.img_labels.iloc[idx]['Pathologie_cardiaque_3_new']

    if self.transform:
        image = self.transform(image)
    if self.target_transform:
        label = self.target_transform(label)

    return image_cropped, label
