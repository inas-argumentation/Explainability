import os.path
import warnings
import torch
import sys
from tqdm import tqdm
from .auxiliary.split_sample import split_sample
from .models_and_data import load_model_tokenizer_dataset_collatefn
from transformers import get_constant_schedule_with_warmup
from .settings import num_training_epochs, base_dir, Config

def update_loss_avg(new_loss, average):
    if average is None:
        average = new_loss
    else:
        average = 0.9*average + 0.1*new_loss
    return average

def evaluate_model(model, tokenizer, dataset, name="evaluation"):
    print(f"\nCalculating accuracy on {name.lower()} set:")

    bar = tqdm(desc="Evaluating... Acc: ", total=len(dataset), position=0, leave=True, file=sys.stdout)
    model.eval()
    num_predictions = 0
    num_correct = 0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        for i in range(len(dataset)):
            sample = dataset[i]
            bar.update(1)

            # Split sample into smaller parts if it is too long to be processed at once
            words, num_splits, split_counts = split_sample(tokenizer, sample[0])
            splits_words = [["".join(w["tokens"]) for w in words if j in w["splits"]] for j in range(num_splits)]

            sample_tokenized = tokenizer(splits_words, return_tensors='pt', truncation=False, is_split_into_words=True, padding=True).to("cuda")

            # The overall prediction is the mean of all predictions on the separate parts
            prediction = torch.mean(torch.sigmoid(model(**sample_tokenized)["logits"]), dim=0)
            prediction_discrete = (prediction.detach().cpu().numpy() > 0.5).astype("int64")

            num_predictions += 1
            num_correct += 1 if prediction_discrete[0] == sample[1] else 0

            bar.desc = f"Evaluating... Acc: {num_correct/num_predictions:<.3f}"


    bar.close()
    acc = num_correct/num_predictions
    print(f"{name} Accuracy: {acc:<.3f}")
    return acc

def train():
    model, tokenizer, dataset, train_dataloader = load_model_tokenizer_dataset_collatefn(load_weights=False, parallel=True)

    print("Start training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=0.05)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    train_dataloader.dataset.set_split("train")
    dataset.set_split("val")
    max_acc = evaluate_model(model, tokenizer, dataset)

    evals_without_improvement = -3
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        for epoch in range(num_training_epochs):
            print(f"\n\nEpoch {epoch}:")
            model.train()
            loss_avg = None
            bar = tqdm(desc="Loss: None", total=len(train_dataloader), position=0, leave=True, file=sys.stdout)

            for idx, batch in enumerate(train_dataloader):
                bar.update(1)
                prediction = model(**batch)

                loss = loss_fn(prediction["logits"], batch["labels"])
                loss.backward()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                loss_avg = update_loss_avg(loss, loss_avg).detach().cpu().numpy()
                bar.desc = f"Loss: {loss_avg:<.3f}"

            bar.close()
            print()

            acc = evaluate_model(model, tokenizer, dataset)

            evals_without_improvement += 1
            if acc >= max_acc:
                print("Model saved.")
                torch.save(model.state_dict(), os.path.join(base_dir, f"saved_models/clf_{Config.save_name}.pkl"))
                max_acc = acc
                evals_without_improvement = (min(evals_without_improvement, 0))

            if evals_without_improvement == 4:
                break