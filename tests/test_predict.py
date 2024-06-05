import pytest
import torch
from perspectives_score_predictor.predict import predict_scores
from utils.data import get_image_filepaths


class DummyModel(torch.nn.Module):
    """
    A dummy PyTorch model for testing purposes.

    This model generates random outputs of size (batch_size, 2).

    Attributes:
        None
    """
    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *).

        Returns:
            torch.Tensor: Random output tensor of shape (batch_size, 2).
        """
        return torch.randn(x.size(0), 2)  # Dummy output for testing


@pytest.fixture
def dummy_data_path(tmp_path):
    """
    Fixture for dummy data path.

    This fixture provides a dummy data path for testing purposes.

    Args:
        tmp_path: Pytest fixture for creating temporary directories.

    Returns:
        str: Path to the dummy data file.
    """
    return "tests/data/0a1d0d53-eaa4-4f42-9ea7-2197bd183520.jpg"


def test_predict_scores(dummy_data_path):
    """
    Test function for predict_scores.

    This function tests the predict_scores function.

    Args:
        dummy_data_path (str): Dummy path to the test data.

    Returns:
        None
    """
    model = DummyModel()
    data_path = dummy_data_path
    outputs = predict_scores(model, data_path)
    assert isinstance(outputs, dict)
    assert len(outputs) == 1

    # Check if all images are processed
    assert dummy_data_path in outputs


