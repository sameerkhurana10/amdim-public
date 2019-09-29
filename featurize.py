from PIL import Image
import numpy as np
import os
import json
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import argparse

from checkpoint import Checkpointer
import kaldi_io as kio

INTERP = 3


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("places_json", help="Path to the json data file for places audio")
    parser.add_argument("img_as_feats_scp", help="Path to the image scp dataset")
    return parser


class ImageDataset(Dataset):
    def __init__(self, dataset_json_file):
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)
        self.data = data_json['data']
        post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize(146, interpolation=INTERP),
            transforms.CenterCrop(128),
            post_transform
        ])

    def _load_image(self, impath):
        img = Image.open(impath).convert('RGB')
        img = self.test_transform(img)
        return img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = self._load_image(item['image'])
        # convert 3D to 2D tensor to store in kaldi-format
        uttid = item["uttid"]
        return {"uttid": uttid, "image": img}


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    # Restore model from the checkpoint
    ckpt = Checkpointer()
    ckpt.restore_model_from_checkpoint(cpt_path="amdim_ndf256_rkhs2048_rd10.pth")
    ckpt.model.to('cuda')
    img_tmp_ark = os.path.splitext(args.img_as_feats_scp)[0] + '.tmp.ark'
    ds = ImageDataset(args.places_json)
    with kio.open_or_fd(img_tmp_ark, 'wb') as f:
        for i in tqdm(range(len(ds))):
            item = ds[i]
            feats = item["image"]
            batch = torch.zeros(2, 3, 128, 128)
            batch[0] = feats
            batch = batch.to('cuda')
            res_dict = ckpt.model(x1=batch, x2=batch, class_only=True)
            global_feats = res_dict["rkhs_glb"][:1]
            k = item["uttid"]
            kio.write_mat(f, global_feats.cpu().detach().numpy(), key=k)
