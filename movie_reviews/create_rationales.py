import os
import numpy as np
from os.path import exists
from .models_and_data import load_model_tokenizer_dataset_collatefn
from .settings import interpretability_approaches, base_dir, Config
from .create_rationale_MaRC import create_rationale
from .create_rationale_captum import create_rationale_captum, set_interpretability_approach
from transformers.utils import logging

# Suppress annoying warnings.
logging.set_verbosity(40)

# Create rationales for a single interpretability method and for the whole dataset split.
def create_rationales_for_complete_data_set(method="MaRC", split="test"):
    if split not in ["train", "val", "test"]:
        raise Exception("split parameter should be one of [\"train\", \"val\", \"split\"]")
    if not exists(os.path.join(base_dir, f"predictions/{Config.save_name}")): os.mkdir(os.path.join(base_dir,f"predictions/{Config.save_name}/"))
    if not exists(os.path.join(base_dir, f"predictions/{Config.save_name}/{method}")):
        os.mkdir(os.path.join(base_dir, f"predictions/{Config.save_name}/{method}"))

    model, tokenizer, dataset, train_dataloader = load_model_tokenizer_dataset_collatefn(parallel=False, load_weights=True)
    dataset.set_split(split)

    if method != interpretability_approaches[0]:
        set_interpretability_approach(method, model)

    for index, sample in [(i, dataset[i]) for i in range(len(dataset))]:
        id = sample[2]["annotation_id"]
        print(f"Generating rationale for sample {index}/{len(dataset)} ({id})...")
        if not exists(os.path.join(base_dir, f"predictions/{Config.save_name}/{method}/{id}.npy")):
            if method == interpretability_approaches[0]:
                weights = create_rationale(model, tokenizer, sample, access_to_gt=True, print_progress=True)
            else:
                weights = create_rationale_captum(sample, tokenizer)
            np.save(os.path.join(base_dir, f"predictions/{Config.save_name}/{method}/{id}.npy"), weights)

def create_rationales_for_complete_data_set_and_all_interpretability_approaches(split="test"):
    for approach in interpretability_approaches:
        print(f"Predicting masks for {approach} approach")
        create_rationales_for_complete_data_set(approach, split)