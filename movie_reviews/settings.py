import os

model_checkpoint = "bert-base-uncased"
batch_size = 8
num_training_epochs = 20
interpretability_approaches = ["MaRC", "Occlusion", "Saliency_L1", "Saliency_L2", "Saliency_Sum", "InputXGrad_L1", "InputXGrad_L2", "InputXGrad_Sum",
                          "LIME", "Integrated_Gradients_L1", "Integrated_Gradients_L2", "Integrated_Gradients_Sum", "Shapley"]
base_dir = os.path.dirname(__file__)

class Config:
    save_name = None
    legacy_mode = False

# Changes to the save_name made using this method will be reflected in all other parts of the code.
def set_save_name(save_name):
    Config.save_name = save_name

# The legacy mode changes two lines in the evaluation class. In both cases, now >= is used instead of >, which makes more sense, but was
# not done to produce the results from the paper. Changes are minimal, but slightly alter the produced scores if legacy mode is deactivated.
def set_legacy_mode(bool):
    Config.legacy_mode = bool

# A different save name allows generating new masks (e.g., to test different parameters) without overwriting previously generated masks.
set_save_name("MaRC_paper")