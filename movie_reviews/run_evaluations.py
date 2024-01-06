import copy
import os.path
import sys
import numpy as np
import torch
import warnings
from tqdm import tqdm
from transformers import logging
from sklearn.metrics import average_precision_score
from .models_and_data import load_model_tokenizer_dataset_collatefn
from .settings import base_dir, interpretability_approaches, Config
from .auxiliary.split_sample import split_sample

logging.set_verbosity_error()

# Calculates the IoU F1 between a set of predicted and evidence indices.
def calc_IoU_F1(words_dict, evidence_indices, predicted_indices):
    spans_gt = []
    spans_pred = []
    for i in range(len(words_dict)):
        if i in evidence_indices:
            if i - 1 not in evidence_indices:
                spans_gt.append([i, i + 1])
            elif i - 1 in evidence_indices:
                spans_gt[-1][-1] = i + 1
        if i in predicted_indices:
            if i - 1 not in predicted_indices:
                spans_pred.append([i, i + 1])
            elif i - 1 in predicted_indices:
                spans_pred[-1][-1] = i + 1

    def calc_IoU(span_1, span_2):
        intersection = len([x for x in range(*span_1) if x >= span_2[0] and x < span_2[1]])
        union = len(set(list(range(*span_1)) + list(range(*span_2))))
        return intersection / union

    IoU_matrix = np.stack([[calc_IoU(span_1, span_2) for span_2 in spans_pred] for span_1 in spans_gt])
    tp = np.sum(np.max(IoU_matrix, axis=0) >= 0.5)
    IoU_precision = tp / max(len(spans_pred), 1)
    IoU_recall = tp / max(len(spans_gt), 1)
    IoU_F1 = 2 * IoU_precision * IoU_recall / ((IoU_precision + IoU_recall) if (IoU_precision + IoU_recall) != 0 else 1)
    return IoU_F1

# Matches the gt evidence provided by the dataset to the words detected by the BERT tokenizer.
def get_gt_evidence_and_prediction_matching(tokenizer, lines, evidences, prediction):
    words = sum([l.split(" ") for l in lines], [])
    words_tokenized = [tokenizer.tokenize(w) for w in words]

    evidence_indices = set()
    for evidence_list in evidences:
        for evidence in evidence_list:
            for i in range(evidence["start_token"], evidence["end_token"]):
                evidence_indices.add(i)

    # visualize_word_importance([(1 if i in evidence_indices else 0, x) for i, x in enumerate(words)])

    words_dict = {}
    i = 0
    for j, word in enumerate(words_tokenized):
        try:
            words_dict[j] = {"word": word[0], "prediction": prediction[i], "gt": 1 if j in evidence_indices else 0,
                             "num_words": 1}
        except:
            words_dict[j] = {"word": word[0], "prediction": 0, "gt": 1 if j in evidence_indices else 0, "num_words": 1}
        i += 1
        for word_part in word[1:]:
            if word_part[:2] != "##":
                try:
                    p = prediction[i]
                except:
                    p = 0
                words_dict[j]["word"] += word_part
                words_dict[j]["prediction"] = (words_dict[j]["prediction"] * words_dict[j]["num_words"] + p) / (
                            words_dict[j]["num_words"] + 1)
                words_dict[j]["num_words"] += 1
                i += 1
            else:
                words_dict[j]["word"] += word_part[2:]

    return words_dict, evidence_indices

# Determines the optimal percentage of words with highest scores that should be selected as rationale to maximize the "target_score"
def get_optimal_word_selection_percentage_for_sample(tokenizer, sample, prediction, target_score="F1"):
    sample = sample[2]
    lines = sample["lines"]

    words_dict, evidence_indices = get_gt_evidence_and_prediction_matching(tokenizer, lines, sample["evidences"], prediction)

    best_percentage = 0
    best_score = 0

    s = np.sort([x["prediction"] for x in words_dict.values()])[::-1]
    for percentage in np.linspace(0.01, 0.99, 150):
        threshold = s[min(int(np.round(percentage * len(words_dict))), len(words_dict)-1)]

        if Config.legacy_mode:
            predictions = np.array([1 if x["prediction"] > threshold else 0 for x in words_dict.values()])
        else:
            predictions = np.array([1 if x["prediction"] >= threshold else 0 for x in words_dict.values()])

        labels = np.array([x["gt"] for x in words_dict.values()])

        if target_score == "F1":
            tp = np.sum(labels * predictions)
            fp = np.sum((1 - labels) * predictions)
            fn = np.sum(labels * (1 - predictions))
            precision = tp / max((tp + fp), 1)
            recall = tp / max((tp + fn), 1)
            f1 = 2 * precision * recall / ((precision + recall) if (precision + recall) != 0 else 1)
            score = f1
        elif target_score == "IoU":
            predicted_indices = [i for i in range(len(words_dict)) if predictions[i] == 1]
            IoU_F1 = calc_IoU_F1(words_dict, evidence_indices, predicted_indices)
            score = IoU_F1
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                AP = average_precision_score(labels, predictions)
                if np.isnan(AP):
                    AP = 0
            score = AP

        if score > best_score:
            best_score = score
            best_percentage = percentage

    return best_percentage

# Do kernel regression predictions. Indices in "leave_out" will be omitted to enable predictions for test set samples without ground truth influence.
def predict_KR(train_data, input, kernel_width, leave_out=[]):
    train_x = np.delete(train_data[:, 0], leave_out)
    train_y = np.delete(train_data[:, 1], leave_out)
    weights = np.exp(-np.square(train_x - input) / kernel_width)
    prediction = np.sum(train_y * weights)/np.sum(weights)
    return prediction

# Generates a dataset that matches the number of words with a score higher than "threshold" to the optimal percentage of
# words that need to be selected to maximize the "target_score".
def generate_kernel_regression_dataset(tokenizer, dataset, method, threshold=0.1, target_score="F1"):
    KR_data = []
    for i in tqdm(range(len(dataset)), desc="Generating kernel regression data", file=sys.stdout):
        sample = dataset[i]

        try:
            mask = np.load(os.path.join(base_dir, f"predictions/{Config.save_name}/{method}/{sample[2]['annotation_id']}.npy"))
        except:
            print()
        y = get_optimal_word_selection_percentage_for_sample(tokenizer, sample, mask, target_score=target_score)

        x = np.sum(mask > threshold)
        KR_data.append([x/len(mask), y])
    return np.array(KR_data)

# Calculate the different scores. Kernel regression is used to predict the optimal percentage of words to be included in the rationale.
def evaluate_prediction_KR(tokenizer, sample, prediction, train_data_for_KR, threshold, index):
    sample = sample[2]
    lines = sample["lines"]

    # Create ground truth in the correct format
    words_dict, evidence_indices = get_gt_evidence_and_prediction_matching(tokenizer, lines, sample["evidences"], prediction)

    percentage_above_threshold = len([x for x in words_dict.values() if x["prediction"] >= threshold]) / len(words_dict)
    gt_array = np.array([x["gt"] for x in words_dict.values()], dtype="int32")

    if train_data_for_KR is not None:
        # Predict percentage of tokens to be included in one-hot prediction
        KR_prediction = predict_KR(train_data_for_KR, percentage_above_threshold, 0.05, leave_out=[index])
        # Determine corresponding score threshold
        threshold = np.sort([x["prediction"] for x in words_dict.values()])[::-1][int(np.round(KR_prediction*len(words_dict)))]

        # Make selection
        if Config.legacy_mode:
            predictions_discrete = np.array([1 if x["prediction"] > threshold else 0 for x in words_dict.values()])
        else:
            predictions_discrete = np.array([1 if x["prediction"] >= threshold else 0 for x in words_dict.values()])
        predicted_indices = [i for i in range(len(words_dict)) if predictions_discrete[i] == 1]

        # Calculate Token F1 score
        tp = np.sum(gt_array * predictions_discrete)
        fp = np.sum((1 - gt_array) * predictions_discrete)
        fn = np.sum(gt_array * (1 - predictions_discrete))
        precision = tp / max((tp + fp), 1)
        recall = tp / max((tp + fn), 1)
        f1 = 2 * precision * recall / ((precision + recall) if (precision + recall) != 0 else 1)

        # Calculate IoU F1 score
        IoU_F1 = calc_IoU_F1(words_dict, evidence_indices, predicted_indices)
    else:
        precision = recall = f1 = IoU_F1 = -1

    predictions = np.array([x["prediction"] for x in words_dict.values()])

    # Calculate mAP score (which is independent of the one-hot selection)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        AP = average_precision_score(gt_array, predictions)
        if np.isnan(AP):
            AP = 0

    return precision, recall, f1, AP, IoU_F1

# Evaluate IoU F1, Token F1 and mAP for the specified method and for all samples from the given dataset split.
def evaluate_predictions(method="MaRC", threshold=0.1, split="test", target_score="F1"):
    _, tokenizer, dataset, _ = load_model_tokenizer_dataset_collatefn(load_weights=False)
    dataset.set_split(split)

    train_data_for_KR = generate_kernel_regression_dataset(tokenizer, dataset, method, threshold=threshold, target_score=target_score)

    scores = []
    for i in tqdm(range(len(dataset)), desc="Evaluating predictions", file=sys.stdout):
        sample = dataset[i]
        mask = np.load(os.path.join(base_dir, f"predictions/{Config.save_name}/{method}/{sample[2]['annotation_id']}.npy"))
        sample_scores = evaluate_prediction_KR(tokenizer, sample, mask, train_data_for_KR, threshold, i)
        scores.append(sample_scores)

    scores = np.stack(scores)
    stats_mean = [np.mean(scores[:,i][np.logical_not(np.isnan(scores[:,i]))], axis=0) for i in range(5)]
    precision, recall, f1, mAP, IoU_F1 = stats_mean

    print(f"Precision:  {precision:.3f}\nRecall:  {recall:.3f}\nF1:  {f1:.3f}\nmAP:  {mAP:.3f}\nIoU F1:  {IoU_F1:.3f}")

def evaluate_all_interpretability_methods(target_score="F1", split="test"):
    for method in interpretability_approaches:
        print(f"\n{method}")
        evaluate_predictions(method, 0.1, split, Config.save_name, target_score)

def evaluate_faithfulness(method):
    model, tokenizer, dataset, _ = load_model_tokenizer_dataset_collatefn(load_weights=True)
    dataset.set_split("test")
    model.eval()
    warnings.filterwarnings("ignore", category=UserWarning)

    # Added randomness is needed as tiebreaker if many words have the same score
    np.random.seed(0)
    mask_randomness = np.random.randn(10000) * 1e-5

    sufficiency_values = []
    comprehensiveness_values = []

    for i in tqdm(range(len(dataset)), file=sys.stdout, desc="Evaluate faithfulness"):
        text, label, sample = dataset[i]

        mask = np.load(os.path.join(base_dir, f"predictions/{Config.save_name}/{method}/{sample['annotation_id']}.npy"))
        mask = mask + mask_randomness[:len(mask)]
        sorted_scores = np.sort(mask)[::-1]
        #mask = np.random.permutation(mask)

        words, num_splits, split_counts = split_sample(tokenizer, text, additionally_extend_last_split=False)

        for test in ["sufficiency", "comprehensiveness"]:
            current_values = []
            for percentage in range(0, 105, 5):
                words_copy = copy.deepcopy(words)
                if percentage == 0:
                    score_threshold = sorted_scores[0] + 0.001
                elif percentage == 100:
                    score_threshold = sorted_scores[-1] - 0.001
                else:
                    score_threshold = sorted_scores[int(np.round(len(sorted_scores)*(percentage/100)))]

                # Remove masked words and create new input text
                for j in range(len(mask)):
                    if (mask[j] <= score_threshold and test == "sufficiency") or (mask[j] > score_threshold and test == "comprehensiveness"):
                        words_copy[j]["tokens"] = ["[PAD]"] * len(words_copy[j]["tokens"])
                input_texts = [" ".join(["".join(words_copy[n]["tokens"]) for n in range(len(mask)) if s in words_copy[n]["splits"]]) for s in range(num_splits)]

                # Add pad tokens to make all splits have same length
                max_length = max(split_counts.values())
                for j in range(len(input_texts)):
                    input_texts[j] += " [PAD]" * (max_length-split_counts[j])

                input_texts_tokenized = tokenizer(input_texts, return_tensors='pt', truncation=False).to("cuda")
                prediction = torch.mean(torch.sigmoid(model(**input_texts_tokenized)["logits"]))
                current_values.append(prediction.detach().cpu().numpy())

            current_values = [label * x + (1-label) * (1-x) for x in current_values]
            if test == "sufficiency":
                sufficiency_values.append(np.mean([current_values[-1] - c for c in current_values[1:-1]]))
            elif test == "comprehensiveness":
                comprehensiveness_values.append(np.mean([current_values[0] - c for c in current_values[1:-1]]))
    sufficiency = np.mean(sufficiency_values)
    comprehensiveness = np.mean(comprehensiveness_values)
    print(f"Sufficiency: {sufficiency:.3f}")
    print(f"Comprehensiveness: {comprehensiveness:.3f}")