# Three different model types were tested. VGG was not included in the paper, as ResNet already represented convolutional networks.
model_types = ["VGG16", "ResNet101", "Vit_b_16"]

# Possible explainability methods for different models
all_explainability_methods = ["MaRC", "Perturb", "IntGrads", "Saliency", "InputXGrad", "Occlusion", "GuidedBP"]
explainability_methods = {"VGG16": ["MaRC", "Perturb", "IntGrads", "Saliency", "InputXGrad", "Occlusion", "GuidedBP"],
                          "ResNet101": ["MaRC", "Perturb", "IntGrads", "Saliency", "InputXGrad", "Occlusion", "GuidedBP"],
                          "Vit_b_16": ["MaRC", "Perturb", "IntGrads", "Saliency", "InputXGrad", "Occlusion"]}

input_image_size = {"VGG16": 224,
                    "ResNet101": 224,
                    "Vit_b_16": 384}

optimization_parameters = {"VGG16": {"w_softmax": 0.9,
                                "w_sigmoid": 0.1,
                                "w_sigma": 1.2,
                                "w_tv": 10,
                                "w_sparsity": 0.6,
                                "w_positive_class": 1,
                                "w_negative_class": 1,
                                "weight_init": 0.5,
                                "sigma_init": 1.2,
                                "backgrounds": ["black", "white", "zero"],
                                "kernel_size": 20},
                           "ResNet101": {"w_softmax": 0.9,
                                "w_sigmoid": 0.1,
                                "w_sigma": 1.2,
                                "w_tv": 10,
                                "w_sparsity": 0.6,
                                "w_positive_class": 1,
                                "w_negative_class": 1,
                                "weight_init": 0.5,
                                "sigma_init": 1.2,
                                "backgrounds": ["black", "white", "zero"],
                                "kernel_size": 20},
                           "Vit_b_16": {"w_softmax": 1,
                                "w_sigmoid": 0,
                                "w_sigma": 1.2,
                                "w_tv": 10,
                                "w_sparsity": 0.25,
                                "w_positive_class": 1,
                                "w_negative_class": 1,
                                "weight_init": 0.5,
                                "sigma_init": 1.2,
                                "backgrounds": ["blur"],
                                "kernel_size": 20}}
# Add universal optimization parameters
optimization_parameters = {x: {**y, **{
                        "kernel_size": 20,
                        "num_optimization_steps": 7500
                            }} for x, y in optimization_parameters.items()}

# Number of epochs for MaRC optimization.
num_optimization_steps = 7500

import os
base_dir = os.path.dirname(__file__)

class Config:

    model_type = None
    save_name = None
    optimization_parameters = None
    input_image_size = None
    explainability_methods = None

# Changes made using this method to the Config class will be recognized by all other parts of the code.
def set_model_type(type):
    Config.model_type = type
    Config.explainability_methods = explainability_methods[type]
    Config.optimization_parameters = optimization_parameters[type]
    Config.input_image_size = input_image_size[type]

# Changes made using this method to the Config class will be recognized by all other parts of the code.
def set_save_name(save_name):
    Config.save_name = save_name

# Change arguments here to run different experiments.
set_model_type(model_types[1])
# A different save name allows generating new masks (e.g., to test different parameters) without overwriting previously generated masks.
set_save_name("MaRC_paper")