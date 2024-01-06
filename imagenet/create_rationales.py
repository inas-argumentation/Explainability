import os.path
import numpy as np
from .models_and_data import ImageDataset, load_model
from .create_rationale_MaRC import create_rationale_MaRC
from .create_rationale_captum import create_rationale_captum
from .create_rationale_pertubations import create_mask_pertubations
from os.path import exists
from .settings import base_dir, Config, all_explainability_methods

def generate_masks_for_whole_dataset(method, split="figure", plot=False):
    dataset = ImageDataset(split)
    model = load_model()

    if not exists(os.path.join(base_dir, f"predictions/{Config.save_name}/{Config.model_type}/{split}/{method}")):
        os.makedirs(os.path.join(base_dir, f"predictions/{Config.save_name}/{Config.model_type}/{split}/{method}"))

    i = 0
    while i < len(dataset):
        sample = dataset[i]
        i += 1
        if exists(os.path.join(base_dir, f"predictions/{Config.save_name}/{Config.model_type}/{split}/{method}/{sample['name']}.npy")):
            continue
        print(f"\nPredicting mask for image {i}/{len(dataset)} ({sample['name']})...")

        if method == all_explainability_methods[0]:
            mask = create_rationale_MaRC(model, dataset, image_dict=sample, plot=plot)
        elif method == all_explainability_methods[1]:
            mask = create_mask_pertubations(model, dataset, image_dict=sample, plot=plot)
        elif method in all_explainability_methods[2:]:
            mask = create_rationale_captum(model, dataset, image_dict=sample, plot=plot, method=method)
        np.save(os.path.join(base_dir, f"predictions/{Config.save_name}/{Config.model_type}/{split}/{method}/{sample['name']}.npy"), mask)

# Generate masks for all samples from the given dataset split and for all methods implemented in this repo.
def generate_masks_for_whole_dataset_and_all_methods(split="figure"):
    for method in Config.explainability_methods:
        print(f"\nPredicting masks for {method} method\n")
        generate_masks_for_whole_dataset(method, split, False)