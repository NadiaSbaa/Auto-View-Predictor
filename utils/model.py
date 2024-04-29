import torch
from utils.shared import BEST_MODEL
from perspectives_score_predictor.model import get_model


def load_model(checkpoint, backbone):
    """
    Loads a pre-trained model from a checkpoint file.

    Args:
        checkpoint (str): Path to the checkpoint file.
        backbone (str): Name of the backbone architecture for the model.

    Returns:
        torch.nn.Module: The loaded model with weights from the checkpoint.
    """
    model = get_model(backbone)
    model.load_state_dict(torch.load(checkpoint))
    return model


def get_best_model():
    """
    Loads the best pre-trained model based on some predefined settings.

    Returns:
        torch.nn.Module: The best pre-trained model.
    """
    best_model = load_model(BEST_MODEL["path"], BEST_MODEL["backbone"])
    return best_model
