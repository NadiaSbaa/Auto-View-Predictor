# Perspectives Score Predictor
The Perspective Score Predictor is an application that provides a simple endpoint for predicting perspective scores on car images.
It uses a pre-trained model to analyze image data and return predicted scores based on various perspectives.

## Training
The model used in this application was inspired by the following papers:
- [An Evaluation of Pre-trained Models for Feature Extraction in Image Classification](https://arxiv.org/pdf/2310.02037)
- [Backbones-Review: Feature Extraction Networks for Deep Learning and Deep Reinforcement Learning Approaches](https://arxiv.org/pdf/2206.08016)

## Architecture
The architecture includes the following:
- **Backbone for Feature Extraction**: A feature extraction block built on popular backbones, known for their effectiveness in feature extraction tasks.
- **Regression Block**: A regression block was chosen for its ability to predict continuous values, which is suitable for predicting perspective scores.

### Backbone 
- **ResNet18**: Residual Network with 18 layers. It introduces skip connections to mitigate the vanishing gradient problem, allowing for deeper networks to be trained effectively.
- **Inception_v3**: A deep convolutional neural network known for its use of inception modules, which allow for efficient feature extraction at multiple scales.
- **EfficientNet_b0**: Part of the EfficientNet family of models, known for their efficient use of parameters and superior performance on resource-constrained devices.
- **ViT_b_16 (Vision Transformer)**: A transformer-based model that applies the transformer architecture to image classification tasks, achieving state-of-the-art performance.


**Backbone Sizes**:

  | Backbone      | Size       |
  |---------------|------------|
  | ResNet18      | 43.175 MB  |
  | Inception_v3  | 22.421 MB  |
  | EfficientNet_b0 | 16.701 MB |
  | ViT_b_16      | 328.049 MB |

## Loss Criterion
The Mean Squared Error (MSE) loss function (`criterion = nn.MSELoss()`) was chosen as it penalizes larger errors more heavily, making it suitable for regression tasks.
It is a common loss function for regression tasks, including multi-label regression tasks.

## Metrics
- **MSE (Mean Squared Error)**: Measures the average squared difference between the predicted and actual values.
- **MAE (Mean Absolute Error)**: Measures the average absolute difference between the predicted and actual values.
- **RMSE (Root Mean Squared Error)**: Measures the square root of the average squared difference between the predicted and actual values.
- **R-squared (R²)**: Represents the proportion of the variance in the dependent variable that is predictable from the independent variables.

## Random Grid Search
A small random grid search was conducted to optimize hyperparameters and improve model performance.
#### Note
A small number of epochs was chosen due to resource limitations. This decision was made to optimize the utilization of available computational resources.
## Evaluation

### Quantitative Evaluation

Quantitative evaluation involves assessing the model's performance using the mentioned metrics (MSE, MAE, RMSE, R-squared) on a separate test dataset.

### Qualitative Evaluation

- Qualitative evaluation involves **visually inspecting predictions** on random data samples to understand how well the model performs in practice. This includes visualizing random images alongside their predicted scores.

- Qualitative evaluation involves **checking predictions distribution** compare the distribution of the predictions with the distribution of actual labels to assess how well the model captures the underlying patterns in the data.

### Model Selection

After evaluation of the ResNet18, Inception_v3, EfficientNet_b0, and ViT_b_16 based models, the ResNet18 backbone based model appears promising and has been selected for further evaluation and tuning.

The regression model seems to perform reasonably well. The MSE, MAE, and RMSE are relatively low, indicating that the model's predictions are close to the actual values. Additionally, the R-squared value of 0.747 indicates a good fit between predicted and actual values, suggesting that the model captures a significant portion of the variance in the data.

However, it's important to note that we need to be careful of potential issues such as data imbalance or bias during further evaluation and tuning.

### Model Optimization Suggestions (To do)
1. **Hyperparameter Tuning**: Conduct a more extensive hyperparameter search to fine-tune the model's performance and consider **increasing the number of epochs** for training.
2. **Data Augmentation**: Rigorous and careful training data Augmentation to improve model generalization.
3. **Transfer Learning**: Explore pre-training domain-specific models for fine-tuning.
4. **Regularization Techniques**: Implement regularization techniques such as dropout or weight decay to prevent overfitting.
5. **Model Compression**: Investigate techniques for model compression to reduce the model size and computational overhead.
6. **Hardware Acceleration**: Utilize hardware accelerators such as GPUs or TPUs to speed up model training and inference.


## Logging

The application logs important events to a file named `api.log` located in the project directory. The log file captures requests received, validation errors, and predictions made.

