import numpy as np
import torch
import matplotlib.pyplot as plt
from .settings import Config
from torch.nn.functional import pad

# Calculates total variation loss (difference between neighboring mask values)
def calculate_tv_loss(image, d, padding):
    tv_loss = 0
    for dir in [(d, 0), (0, d), (d, d), (-d, d)]:
        tv_loss += torch.square(image[padding[1][0] + max(0, dir[0]) : Config.input_image_size - padding[1][1] + min(0, dir[0]),
                                padding[0][0] + max(0, dir[1]) : Config.input_image_size - padding[0][1] + min(0, dir[1])] -
                                image[padding[1][0]-min(0, dir[0]) : Config.input_image_size - padding[1][1] - max(0, dir[0]),
                                padding[0][0] - min(0, dir[1]) : Config.input_image_size - padding[0][1] - max(0, dir[1])]).sum()
    return tv_loss

def create_rationale_MaRC(model, dataset, image_dict, plot=False):
    model.eval()

    parameters = Config.optimization_parameters
    n_backgrounds = len(parameters["backgrounds"])

    original_image = image_dict["image_tensor"].repeat(n_backgrounds, 1, 1, 1)
    label = image_dict["label"]
    padding = image_dict["padding_tensor"]

    backgrounds = []
    if "white" in parameters["backgrounds"]:
        backgrounds.append(dataset.get_white_image())
    if "black" in parameters["backgrounds"]:
        backgrounds.append(dataset.get_black_image())
    if "zero" in parameters["backgrounds"]:
        backgrounds.append(dataset.get_mean_image())
    if "blur" in parameters["backgrounds"]:
        backgrounds.append(dataset.get_blurred_image(image_dict["image_PIL_resized"], radius=20)[0])
    backgrounds = torch.cat(backgrounds, dim=0)

    # Define optimization variables that are used to calculate mask values
    weights = torch.tensor(np.ones((Config.input_image_size - (sum(padding[1])), Config.input_image_size - (sum(padding[0]))), dtype="float32") * parameters["weight_init"], requires_grad=True, device="cuda", dtype=torch.float32)
    sigmas = torch.tensor(np.ones((Config.input_image_size - (sum(padding[1])), Config.input_image_size - (sum(padding[0]))), dtype="float32") * parameters["sigma_init"], requires_grad=True, device="cuda", dtype=torch.float32)

    # One-hot mask that indicates which pixels actually belong to the image (not to the padding)
    one_hot_multipliers = torch.ones((Config.input_image_size - (sum(padding[1])), Config.input_image_size - (sum(padding[0]))), dtype=torch.float32)
    one_hot_multipliers = pad(one_hot_multipliers.unsqueeze(0), (padding[0][0], padding[0][1], padding[1][0], padding[1][1]), mode="constant").squeeze(0).to("cuda")
    num_valid_pixels = torch.sum(one_hot_multipliers)

    optimizer = torch.optim.Adam([weights, sigmas], lr=3e-2)

    # Creates distance grid that is used to weight influences of weights from neighboring weights
    grid = torch.arange(start=0, end=2*parameters["kernel_size"]+1, step=1).repeat(2*parameters["kernel_size"]+1, 1).to("cuda")
    x_y_grid = torch.stack([grid, torch.transpose(grid, 0, 1)], dim=-1)
    distances = (x_y_grid - x_y_grid[parameters["kernel_size"], parameters["kernel_size"], :]).square().sum(dim=-1).repeat(Config.input_image_size, Config.input_image_size, 1, 1)

    last_weight_mean = 1
    weight_mean_diff_exp = None

    for e in range(parameters["num_optimization_steps"]):
        # Add padding to optimization parameters to match input image size
        sigmas_pad = pad(sigmas.unsqueeze(0), (parameters["kernel_size"]+padding[0][0], parameters["kernel_size"]+padding[0][1],
                                               parameters["kernel_size"]+padding[1][0], parameters["kernel_size"]+padding[1][1]), mode="replicate").squeeze(0)
        weights_pad = pad(weights.unsqueeze(0), (parameters["kernel_size"]+padding[0][0], parameters["kernel_size"]+padding[0][1],
                                                 parameters["kernel_size"]+padding[1][0], parameters["kernel_size"]+padding[1][1]), mode="replicate").squeeze(0)

        # Calculate current mask values
        sigmas_unfold = sigmas_pad.unfold(0, 2*parameters["kernel_size"]+1, 1).unfold(1, 2*parameters["kernel_size"]+1, 1)
        weights_unfold = weights_pad.unfold(0, 2*parameters["kernel_size"]+1, 1).unfold(1, 2*parameters["kernel_size"]+1, 1)
        current_mask_values = torch.sigmoid((torch.exp(-distances / sigmas_unfold) * weights_unfold).sum(dim=-1).sum(dim=-1)) * one_hot_multipliers

        # Apply mask to image
        input_image = current_mask_values * original_image + (1-current_mask_values) * backgrounds
        complement_image = (1-current_mask_values) * original_image + current_mask_values * backgrounds

        # Calculate predicted class probabilities
        output = model(torch.cat([input_image, complement_image], dim=0))
        output_prob_softmax = torch.softmax(output, dim=-1) * 0.99999 + 0.000005
        output_prob_sigmoid = torch.sigmoid(output) * 0.99999 + 0.000005
        output_prob = (parameters["w_sigmoid"] * output_prob_sigmoid + parameters["w_softmax"] * output_prob_softmax)

        # Calculate the different parts of the loss function
        tv_loss = parameters["w_tv"] * (1 * calculate_tv_loss(current_mask_values, 1, padding)) / num_valid_pixels
        class_scoring_loss = - (parameters["w_positive_class"] * torch.log(output_prob[:n_backgrounds, label])).mean()
        complement_class_scoring_loss = - (parameters["w_negative_class"] * torch.log(1 - output_prob[n_backgrounds:, label])).mean()
        sigma_reg_loss = - parameters["w_sigma"] * torch.log(sigmas).mean()
        weights_reg_loss = parameters["w_sparsity"] * torch.square(current_mask_values.sum()/num_valid_pixels)

        total_loss = class_scoring_loss + complement_class_scoring_loss + sigma_reg_loss + weights_reg_loss + tv_loss
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()

        # Sigma values must not get too small
        with torch.no_grad():
            sigmas.clip_(min=0.6)

        # Track the difference in average mask values to check for convergence
        current_weight_mean = (current_mask_values.sum()/num_valid_pixels).detach().cpu().numpy()
        weight_mean_diff = last_weight_mean - current_weight_mean
        last_weight_mean = current_weight_mean
        # Calculate exponential moving average to be less susceptible to individual noisy steps
        weight_mean_diff_exp = 0.95*weight_mean_diff_exp + 0.05*weight_mean_diff if weight_mean_diff_exp is not None else weight_mean_diff

        # Check stop conditions
        if weight_mean_diff_exp < 0.0002 and e >= 250 and current_weight_mean < 0.5:
            break
        if e % 50 == 0:
            print(f"Step {e}\nClass scores: {output_prob[:n_backgrounds, label]}\nComplement class scores: {output_prob[n_backgrounds:, label]}")
            print(f"Weights: {current_weight_mean}\n")

    if plot:
        result_weights = np.expand_dims(current_mask_values.np(), axis=-1)
        resulting_masked_image = np.asarray(result_weights * np.asarray(image_dict["image_PIL_resized"], dtype="float32") +
                                            (1-result_weights) * np.zeros((Config.input_image_size, Config.input_image_size, 3), dtype="float32"), dtype="uint8")

        f, axarr = plt.subplots(1, 3)
        axarr[0].imshow(image_dict["image_PIL_resized"])
        axarr[1].imshow(np.asarray(result_weights*255, dtype="uint8"))
        axarr[2].imshow(resulting_masked_image)
        plt.show()

    return current_mask_values.np()