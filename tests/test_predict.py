import pytest
import torch
from perspectives_score_predictor.predict import predict_scores
from utils.data import get_image_filepaths


class DummyModel(torch.nn.Module):
    def forward(self, x):
        return torch.randn(x.size(0), 2)  # Dummy output for testing


@pytest.fixture
def dummy_data_path(tmp_path):
    return "data/test/0a1d0d53-eaa4-4f42-9ea7-2197bd183520.jpg"


def test_predict_scores(dummy_data_path):
    model = DummyModel()
    data_path = dummy_data_path
    outputs = predict_scores(model, data_path)
    assert isinstance(outputs, dict)
    assert len(outputs) == 1

    # Check if all images are processed
    assert dummy_data_path in outputs


