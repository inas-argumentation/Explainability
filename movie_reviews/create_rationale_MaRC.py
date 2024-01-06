import warnings
import torch
import numpy as np
from movie_reviews.auxiliary.visualize_text import visualize_word_importance
from movie_reviews.auxiliary.split_sample import split_sample

def to_np(self):
    return self.detach().cpu().numpy()
setattr(torch.Tensor, "np", to_np)

num_optimization_steps = 900
random_change_percentage = 0.05
weights_loss_factor = 1
sigma_loss_factor = 1.2

def print_prediction_values(tensor, selection, split_set):
    values_np = tensor.detach().cpu().numpy()
    num_splits = int(values_np.shape[0]/2)
    print("split | value | complement value")
    for i, s in zip(range(num_splits), split_set):
        print(f"  {s}   | {float(values_np[i][selection]):.3f} |     {float(values_np[num_splits+i][selection]):.3f}")

# Set multi_label=True if the model outputs multiple logits per sample for multiclass classification (with softmax).
# Set multi_label=False if the model outputs a single logit per sample for binary classification (sigmoid)
# "models" can be a single model or a list of models (they have to use the same tokenizer). In the latter case, the mask is created to fit all models, which can reduce overfitting to a single model.
# The input is split into multiple parts if longer than 510 tokens. The masks for the splits are calculated in parallel. Set max_number_of_parallel_splits to
# limit the number of parallel splits if GPU memory does not allow optimizing all of them in parallel.
def create_rationale(models, tokenizer, sample, multi_label=False, access_to_gt=True, print_progress=True, gt_indices=None, max_number_of_parallel_splits=3):
    if type(models) != list:
        models = [models]
    with (warnings.catch_warnings()):
        warnings.filterwarnings("ignore", category=UserWarning)

        for i in range(len(models)):
            models[i].eval()
        words, num_splits, split_word_counts = split_sample(tokenizer, sample[0])

        # Lists how many tokens each word has. A token with count -1 will not be optimized (e.g., PAD, CLS).
        split_word_tokens_counts = [[-1] for _ in range(num_splits)]
        split_words = [list() for _ in range(num_splits)]
        for word in words:
            for split in word["splits"]:
                split_word_tokens_counts[split].append(word["num_tokens"])
                split_words[split].append(word["word"])

        # Append [PAD] tokens to split_words[split] list to make all splits have the same number of tokens.
        max_split_length = max(split_word_counts.values())
        num_pads_per_split = []
        for i in range(num_splits):
            if (diff := max_split_length - split_word_counts[i]) > 0:
                split_word_tokens_counts[i] = split_word_tokens_counts[i] + [-1] * diff
                split_words[i] = split_words[i] + ["[PAD]"] * diff
            num_pads_per_split.append(diff)

        all_word_tokens_counts = []
        for i in range(num_splits):
            split_word_tokens_counts[i].append(-1)
            all_word_tokens_counts = all_word_tokens_counts + split_word_tokens_counts[i]

        sample_tokenized = tokenizer(split_words, return_tensors='pt', truncation=False, is_split_into_words=True)
        # Use sequence of PAD token embeddings as uninformative input
        uninformative_input = tokenizer([("[PAD] " * max_split_length)[:-1]]*num_splits, return_tensors='pt', truncation=False)

        embeddings_sample = [m.bert.embeddings(sample_tokenized["input_ids"].to("cuda")) for m in models]
        embeddings_uninformative = [m.bert.embeddings(uninformative_input["input_ids"].to("cuda")) for m in models]

        label = sample[1]

        # The parameters to be optimized
        weights = [torch.tensor([1.2], requires_grad=True, device="cuda", dtype=torch.float32) if r != -1 else
                   torch.tensor([-20], requires_grad=False, device="cuda", dtype=torch.float32) for r in all_word_tokens_counts]
        sigmas = [torch.tensor([2.0], requires_grad=True, device="cuda", dtype=torch.float32) if r != -1 else
                  torch.tensor([0.01], requires_grad=False, device="cuda", dtype=torch.float32) for r in all_word_tokens_counts]
        optimizer = torch.optim.AdamW(weights + sigmas, lr=3e-2)

        num_parameters = [len([k for k in r if k != -1]) for r in split_word_tokens_counts]
        split_word_tokens_counts = [[x if x != -1 else 1 for x in r] for r in split_word_tokens_counts]

        probability_func = (lambda x: torch.softmax(x, dim=-1)) if multi_label else torch.sigmoid
        # If the model does not have access to the ground truth, the optimization is done for the predicted label.
        if not access_to_gt:
            predictions = [torch.mean(probability_func(m(inputs_embeds=e, attention_mask=torch.ones((num_splits, max_split_length + 2)).to("cuda"))["logits"]), dim=0) for m, e in zip(models, embeddings_sample)]
            if multi_label:
                prediction = torch.stack(predictions, dim=0).mean(0)
                label = int(np.argmax(prediction.np(), axis=-1))
            else:
                prediction = torch.tensor(predictions).mean()
                label = int(np.round(prediction.np()))

        print(f"Optimizing for label {label}")

        split_indices = [list(range(num_splits))[i:i+max_number_of_parallel_splits] for i in range(0, num_splits, max_number_of_parallel_splits)]
        final_weights = {}

        prev_split_end_prev_set = 0
        for split_set in split_indices:
            last_mask_means = [1 for _ in split_set]  # Track the average mask value to check for stop conditions
            attention_mask = torch.ones((2 * len(split_set), max_split_length + 2)).to("cuda")

            for i in range(num_optimization_steps):
                mask_tensors = []

                # Store sigmas and mask values for regularization loss calculation. These mask values are different from "mask_tensors", as values are not repeated for words with multiple tokens.
                all_individual_mask_values = []
                all_individual_sigmas = []

                prev_split_end = prev_split_end_prev_set
                for j in split_set: # Calculate mask values from weights and sigmas
                    current_num_words = len(split_word_tokens_counts[j])

                    sigmas_tensor = torch.cat(sigmas[prev_split_end:prev_split_end+current_num_words])
                    all_individual_sigmas.append(sigmas_tensor)
                    weights_tensor = torch.cat(weights[prev_split_end:prev_split_end+current_num_words])
                    prev_split_end = prev_split_end + current_num_words

                    distance_values = (torch.arange(start=0, end=current_num_words, step=1).repeat((current_num_words, 1)) -
                                   torch.unsqueeze(torch.arange(start=0, end=current_num_words, step=1), -1)).to("cuda")
                    distance_values = torch.square(distance_values) / torch.square(torch.unsqueeze(sigmas_tensor, dim=-1))
                    mask_values = torch.sigmoid((torch.exp(-distance_values) * torch.unsqueeze(weights_tensor, dim=-1)).sum(dim=0))
                    all_individual_mask_values.append(mask_values[1:-(1+num_pads_per_split[j])])

                    mask_values_repeated = [mask_values[k].repeat(split_word_tokens_counts[j][k]) for k in range(len(split_word_tokens_counts[j]))]
                    mask_values_repeated = torch.unsqueeze(torch.unsqueeze(torch.cat(mask_values_repeated, dim=0), dim=-1), dim=0)
                    mask_tensors.append(mask_values_repeated)
                mask_tensors = torch.squeeze(torch.stack(mask_tensors, dim=0), dim=1)

                # Randomly set mask values to 0 or 1
                ones = torch.ones_like(mask_tensors, device="cuda", dtype=torch.float32)
                d_1 = (torch.empty_like(mask_tensors, device="cuda").uniform_() > random_change_percentage).type(torch.float32) # Select values to set to 0
                d_2 = (torch.empty_like(mask_tensors, device="cuda").uniform_() > random_change_percentage).type(torch.float32) # Select values to set to 1
                both = (1-d_1)*(1-d_2) # If a word is selected by both, do not change it.
                d_1 = d_1 + both * ones
                d_2 = d_2 + both * ones
                mask_tensors = mask_tensors * d_1 * d_2 + ones * (1-d_2)

                classification_loss = torch.zeros((1,), device="cuda", dtype=torch.float32)
                all_predictions = []
                for m_idx in range(len(models)):
                    masked_embeddings = embeddings_sample[m_idx][split_set] * mask_tensors + embeddings_uninformative[m_idx][split_set] * (1-mask_tensors)
                    complement_masked_embeddings = embeddings_sample[m_idx][split_set] * (1-mask_tensors) + embeddings_uninformative[m_idx][split_set] * mask_tensors
                    embeddings = torch.cat([masked_embeddings, complement_masked_embeddings], dim=0)
                    embeddings = embeddings + torch.empty_like(embeddings, device="cuda").normal_(mean=0.0, std=0.03) # Add some Gaussian noise to the input

                    prediction = probability_func(models[m_idx](inputs_embeds=embeddings, attention_mask=attention_mask)["logits"]) * 0.99999 + 0.000005
                    all_predictions.append(prediction)

                    masked_predictions = prediction[:num_splits].mean(0)
                    complement_masked_predictions = prediction[num_splits:].mean(0)
                    if multi_label:
                        masked_loss = -torch.log(masked_predictions[label])
                        complement_masked_loss = -torch.log(1 - complement_masked_predictions[label])
                    else:
                        masked_loss = torch.squeeze(- ((1-label) * torch.log(1-masked_predictions) + label * torch.log(masked_predictions)), dim=-1)
                        complement_masked_loss = torch.squeeze(- ((1-label) * torch.log(complement_masked_predictions) + label * torch.log(1-complement_masked_predictions)), dim=-1)

                    classification_loss += masked_loss.sum() + complement_masked_loss.sum()

                classification_loss = classification_loss / len(models)
                weights_loss = weights_loss_factor * torch.square(mask_means := torch.cat([torch.sum(all_individual_mask_values[j], dim=0, keepdim=True)/num_parameters[j] for j in range(len(split_set))], dim=0))
                sigma_loss = sigma_loss_factor * torch.cat([torch.sum(-torch.log(all_individual_sigmas[j]), dim=0, keepdim=True)/num_parameters[j] for j in range(len(split_set))], dim=0)
                loss = classification_loss + (sigma_loss + weights_loss).sum()

                loss.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()

                prediction = torch.stack(all_predictions, dim=0).mean(0)
                if i == 0 and print_progress:
                    print("\nInitial values:")
                    print_prediction_values(prediction, selection=label if multi_label else 0, split_set=split_set)
                    print("Avg mask values: ", *[f"{x:.3f}" for x in mask_means.detach().cpu().numpy()])
                if i > 0 and i % 50 == 0:
                    mask_means_np = mask_means.detach().cpu().numpy()
                    if print_progress:
                        print(f"\nIteration {i}")
                        print_prediction_values(prediction, selection=label if multi_label else 0, split_set=split_set)
                        print("Avg mask values: ", *[f"{x:.3f}" for x in mask_means_np])
                    for j in range(len(split_set)):
                        diff = abs(last_mask_means[j] - mask_means_np[j])
                        # Check for stop conditions
                        if (diff < 1/200 or (diff < 1/80 and mask_means_np[j] < 0.2)) and (i >= 199 or mask_means_np[j] < 0.3)\
                            and split_set[j] not in final_weights and mask_means_np[j] < 0.45:
                            final_weights[split_set[j]] = all_individual_mask_values[j]
                            if print_progress: print(f"Saved mask for split {j}")
                    last_mask_means = mask_means_np
                    if len([s for s in split_set if s in final_weights]) == len(split_set): # Masks for all splits have been saved
                        break

            # If no stop condition for a split has been reached, take the last mask values as result.
            for j in range(len(split_set)):
                if split_set[j] not in final_weights:
                    final_weights[split_set[j]] = all_individual_mask_values[j]

            prev_split_end_prev_set = prev_split_end

        # If the sample was split into multiple parts, merge individual mask parts by blending overlapping parts linearly
        #result_weight = final_weights[0][:-num_pads_per_split[0]] if num_pads_per_split[0] > 0 else final_weights[0]
        result_weight = final_weights[0]
        for j in range(1, len(final_weights)):
            transition_length = len([w for w in words if j in w["splits"] and j-1 in w["splits"]])
            transition_weight = (torch.arange(start=0, end=transition_length, step=1) / transition_length).to("cuda")
            #tensor = final_weights[j][:-num_pads_per_split[j]] if num_pads_per_split[j] > 0 else final_weights[j]
            tensor = final_weights[j]
            result_weight[-transition_length:] = result_weight[-transition_length:] * (1-transition_weight) + tensor[:transition_length] * transition_weight
            result_weight = torch.cat([result_weight, tensor[transition_length:]])

        if result_weight.shape[0] != len(words):
            raise Exception()
        result_weight = result_weight.detach().cpu().numpy()
        if print_progress:
            if gt_indices is not None:
                visualize_word_importance(list(zip(result_weight,
                                                   [1 if i in gt_indices else 0 for i in range(len(words))],
                                                   [w["word"] for w in words])))
            else:
                visualize_word_importance(list(zip(result_weight, [w["word"] for w in words])))
        return result_weight