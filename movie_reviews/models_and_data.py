import copy
import torch
import json
import random
import numpy as np
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from .settings import batch_size, model_checkpoint, base_dir, Config
from .auxiliary.split_sample import split_sample

def load_data_for_JSONL_file(file):
    with open(os.path.join(base_dir, f"data/{file}"), 'r') as f:
        data = list(f)

    data_dict = {}
    i = 0
    for d in data:
        parsed = json.loads(d)
        with open(os.path.join(base_dir, f"data/docs/{parsed['annotation_id']}"), 'r') as f:
            parsed["lines"] = f.read().split("\n")
            parsed["text"] = " ".join(parsed["lines"])
        parsed["label"] = 1 if parsed["classification"] == 'POS' else 0
        data_dict[i] = parsed
        i += 1
    return data_dict

# The dataset for the movie reviews. The method set_split() can be used to select the corresponding split ("train", "val", "test").
class IMDBDataset(Dataset):
    def __init__(self):

        train_data = load_data_for_JSONL_file("train.jsonl")
        val_data = load_data_for_JSONL_file("val.jsonl")
        test_data = load_data_for_JSONL_file("test.jsonl")

        self.data = {int(x["annotation_id"][5:8])*2 + (1 if x["annotation_id"][0] == "n" else 0): x
                     for x in list(train_data.values()) + list(val_data.values()) + list(test_data.values())}

        train_indices = list(range(1600))
        val_indices = list(range(1600, 1800))
        test_indices = [x for x in range(1800, 2000) if x in self.data]

        self.indices = {"train": train_indices,
            "val": val_indices,
            "test": test_indices
        }

        self.split = None
        self.set_split("train")

    def set_split(self, split):
        self.split = split

    def __len__(self):
        return len(self.indices[self.split])

    def __getitem__(self, idx):
        sample = self.data[self.indices[self.split][idx]]
        return sample["text"], sample["label"], sample

# Collate function for model training. Performs some random data augmentation.
def collate_fn(batch, tokenizer, use_randomness=False):
    texts = [e[0] for e in batch]
    labels = np.array([e[1] for e in batch]).astype("float32")

    all_splits_words = []
    all_labels = []
    for i in range(len(texts)):
        text_tokenized = tokenizer.tokenize(texts[i])

        # Randomly shift the start of the text during training as data augmentation
        random_shift = random.randint(-30, max(min(130, len(text_tokenized)-170), -20))
        if random_shift < 0 or not use_randomness:
            random_shift = 0
        text_tokenized = text_tokenized[random_shift:]
        if text_tokenized[0][:2] == "##":
            text_tokenized[0] = text_tokenized[0][2:]
        j = 1
        while j < len(text_tokenized):
            if text_tokenized[j][:2] == "##":
                text_tokenized[j-1] = text_tokenized[j-1] + text_tokenized[j][2:]
                del text_tokenized[j]
            else:
                j += 1
        text = " ".join(text_tokenized)

        # Split text (if more than 510 tokens) into multiple splits and select a random split for training
        words, num_splits, splits_counts = split_sample(tokenizer, text)
        sample = random.randrange(0, num_splits)
        splits_words = ["".join(w["tokens"]) for w in words if sample in w["splits"]]

        all_splits_words.append(splits_words)
        all_labels.append(labels[i])

    tokenized_inputs = tokenizer(all_splits_words, return_tensors='pt', max_length=512, truncation=False, is_split_into_words=True, padding=True).to("cuda")
    tokenized_inputs["labels"] = torch.unsqueeze(torch.Tensor(all_labels), dim=-1).to("cuda")

    return tokenized_inputs

def load_model_tokenizer_dataset_collatefn(parallel=False, load_weights=True):
    model = torch.nn.DataParallel(AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=1),
                                  device_ids=list(range(torch.cuda.device_count()))).to("cuda")

    if load_weights:
        model.load_state_dict(torch.load(os.path.join(base_dir, f"saved_models/clf_{Config.save_name}.pkl")), strict=False)
    if not parallel:
        model = model.module
    train_collate_fn = (lambda x: collate_fn(x, tokenizer, True))
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    dataset = IMDBDataset()
    train_dataloader = DataLoader(copy.deepcopy(dataset), collate_fn=train_collate_fn, batch_size=batch_size, shuffle=True)
    return model, tokenizer, dataset, train_dataloader