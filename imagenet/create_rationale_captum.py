import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import ReLU
from captum.attr import IntegratedGradients, Occlusion, InputXGradient, Saliency, GuidedBackprop
from .settings import Config

def convert_relu(model):
    for child_name, child in model.named_children():
        if isinstance(child, ReLU):
            setattr(model, child_name, ReLU())
        else:
            convert_relu(child)

def create_rationale_captum(model, dataset, image_dict, plot=False, method="IG"):
    model.eval()
    attr_model = lambda x: torch.softmax(model(x), dim=-1)

    black_image, white_image = dataset.get_black_image(), dataset.get_white_image()
    zero_image = torch.zeros_like(black_image)

    original_image = image_dict["image_tensor"].repeat(1, 1, 1, 1)
    label = image_dict["label"]

    if method == "IntGrads":
        attr = IntegratedGradients(model, multiply_by_inputs=True)
        blurred_image = dataset.get_blurred_image(image_dict["image_PIL_resized"])[0]
        # Different backgrounds can be used. We used a blurred version of the image as it produced good results more regularly than the other methods
        # black_attr = torch.norm(attr.attribute(original_image, baselines=black_image, target=label, n_steps=100, internal_batch_size=2), p=1, dim=1).squeeze(0).detach().cpu().numpy()
        # white_attr = torch.norm(attr.attribute(original_image, baselines=white_image, target=label, n_steps=100, internal_batch_size=2), p=1, dim=1).squeeze(0).detach().cpu().numpy()
        # zero_attr = torch.norm(attr.attribute(original_image, baselines=zero_image, target=label, n_steps=100, internal_batch_size=2), p=1, dim=1).squeeze(0).detach().cpu().numpy()
        # mask = np.maximum(np.maximum(black_attr, white_attr), zero_attr)
        mask = torch.norm(attr.attribute(original_image, baselines=blurred_image, target=label, n_steps=100, internal_batch_size=2), p=1, dim=1).squeeze(0).detach().cpu().numpy()
    elif method == "Saliency":
        attr = Saliency(model)
        mask = torch.norm(attr.attribute(original_image, target=label), p=1, dim=1).squeeze(0).detach().cpu().numpy()
    elif method == "InputXGrad":
        attr = InputXGradient(model)
        mask = torch.norm(attr.attribute(original_image, target=label), p=1, dim=1).squeeze(0).detach().cpu().numpy()
    elif method == "GuidedBP":
        def relu_hook_function(module, grad_in, grad_out):
            if isinstance(module, ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)

        for module in model.modules():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_hook_function)
        attr = GuidedBackprop(model)
        mask = torch.norm(attr.attribute(original_image, target=label), p=1, dim=1).squeeze(0).detach().cpu().numpy()
    elif method == "Occlusion":
        with torch.no_grad():
            attr = Occlusion(attr_model)
            blurred_image = dataset.get_blurred_image(image_dict["image_PIL_resized"])[0]
            window_size = int(np.round(Config.input_image_size/9))
            stride = int(np.round(Config.input_image_size/56))
            # Different backgrounds can be used. We used a blurred version of the image as it produced good results more regularly than the other methods
            # black_attr = torch.norm(attr.attribute(original_image, strides=(3, 4, 4), perturbations_per_eval=2, target=label, sliding_window_shapes=(3, 25, 25), baselines=black_image), p=1, dim=1).squeeze(0).detach().cpu().numpy()
            # zero_attr = torch.norm(attr.attribute(original_image, strides=(3, 4, 4), perturbations_per_eval=2, target=label, sliding_window_shapes=(3, 25, 25), baselines=zero_image), p=1, dim=1).squeeze(0).detach().cpu().numpy()
            # white_attr = torch.norm(attr.attribute(original_image, strides=(3, 4, 4), perturbations_per_eval=2, target=label, sliding_window_shapes=(3, 25, 25), baselines=white_image), p=1, dim=1).squeeze(0).detach().cpu().numpy()
            # mask = np.maximum(np.maximum(black_attr, white_attr), zero_attr)
            mask = torch.norm(attr.attribute(original_image, strides=(3, stride, stride), perturbations_per_eval=2, target=label, sliding_window_shapes=(3, window_size, window_size), baselines=blurred_image), p=1, dim=1).squeeze(0).detach().cpu().numpy()

    else:
        print("Method not available.")
        quit()

    mask = mask / np.max(mask)

    if plot:
        result_weights = np.expand_dims(mask, axis=-1)
        resulting_masked_image = np.asarray(result_weights * np.asarray(image_dict["image_PIL_resized"], dtype="float32") +
                                            (1 - result_weights) * np.zeros((Config.input_image_size, Config.input_image_size, 3), dtype="float32"), dtype="uint8")

        f, axarr = plt.subplots(1, 3)
        axarr[0].imshow(image_dict["image_PIL_resized"])
        axarr[1].imshow(np.asarray(result_weights*255, dtype="uint8"))
        axarr[2].imshow(resulting_masked_image)
        plt.show()

    return mask