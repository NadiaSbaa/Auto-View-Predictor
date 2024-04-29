from math import sqrt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from torch.utils.data import DataLoader

from perspectives_score_predictor.predictionCustomDataset import PredictionCustomDataset


def predict_scores(model, data_path):
    """
    Predicts scores using the given model and data.

    Args:
        model (torch.nn.Module): The model used for prediction.
        data_path (str): Path to the data.

    Returns:
        dict: A dictionary containing the predicted scores for each image.
    """
    dataset = PredictionCustomDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    outputs = {}
    with torch.no_grad():
        for data in dataloader:
            image_filepath, image = data
            output = model(image)
            output = output.cpu().detach().tolist()
            # reformat outputs
            for i in range(len(image_filepath)):
                outputs[image_filepath[i]] = output[i]

    return outputs


def evaluate(model, criterion, dataloader):
    """
    Evaluates the model using the given criterion and dataloader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        criterion: The loss criterion used for evaluation.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the evaluation data.

    Returns:
        tuple: A tuple containing the test losses, labels, and predictions.
    """
    test_losses = []
    test_predictions_all = []
    test_labels_all = []

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            outputs = model(inputs)
            test_loss = criterion(outputs, labels).item()
            test_losses.append(test_loss)
            test_predictions_all.extend(outputs.numpy())
            test_labels_all.extend(labels.numpy())

    # Calculate test metrics
    test_losses = np.array(test_losses)
    test_predictions_all = np.array(test_predictions_all)
    test_labels_all = np.array(test_labels_all)
    test_mse = mean_squared_error(test_labels_all, test_predictions_all)
    test_mae = mean_absolute_error(test_labels_all, test_predictions_all)
    test_rmse = sqrt(test_mse)
    test_r2 = r2_score(test_labels_all, test_predictions_all)
    print('Test Loss: %.3f, MSE: %.3f, MAE: %.3f, RMSE: %.3f, R-squared: %.3f' %
          (np.mean(test_losses), test_mse, test_mae, test_rmse, test_r2))

    return test_losses, test_labels_all, test_predictions_all
