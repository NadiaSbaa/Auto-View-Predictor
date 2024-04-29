# Perspectives Score Predictor


Der Perspective Score Predictor ist eine Anwendung, die einen einfachen Endpunkt zur Vorhersage von Perspektivbewertungen auf Autobildern bereitstellt.
Es verwendet ein vortrainiertes Modell zur Analyse von Textdaten und gibt basierend auf verschiedenen Perspektiven vorhergesagte Bewertungen zurück.
<br> 
The Perspective Score Predictor is an application that provides a simple endpoint for predicting perspective scores on car images.
It uses a pre-trained model to analyze text data and return predicted scores based on various perspectives.

## Funktionen

- **Prediction Endpoint**: Stellt einen Endpunkt zur Vorhersage von Perspektivbewertungen auf Eingabedaten bereit.
- **Datenvalidierung**:Validiert Eingabedaten, um sicherzustellen, dass sie das erforderliche Format und die Kriterien erfüllen.
- **Logging**: Logs Anfragen, Validierungsfehler und Vorhersagen zu Überwachungs- und Debugging-Zwecken.

## Installation

Um den Perspective Score Predictor lokal auszuführen, befolgen Sie diese Schritte:

1. Klone das Repository:

    ```bash
    git clone git@github.com:NadiaSbaa/Perspectives_Score_Predictor.git
    ```

2. Navigiere zum Projektverzeichnis:

    ```bash
    cd Perspectives_Score_Predictor
    ```

3. Installiere dependencies:
     ```bash
    pip install -r requirements.txt
    ```

4. Starte die App:
     ```bash
    flask run
    ```

### Verwendung

#### Vorhersage-Endpunkt
Der Vorhersage-Endpunkt kann über HTTP-POST-Anforderung erreicht werden. Er akzeptiert ein JSON-Objekt, das den Pfad zu den zu analysierenden Daten enthält.

- **GET /predict**:  Vorhersage von Perspektivbewertungen der Eingabedaten

  - **Request Parameters**:
    - `data_path` (string): Bilddateipfad (Format: '.png', '.jpg', '.jpeg', '.gif', '.bmp') oder Verzeichnispfad

  - **Antwort**:
      ```json
      {
    "Predicted scores": {
        "image_path1": [
            perspective_score_hood,
            perspective_score_backdoor_left
        ],
          "image_path2": [
            perspective_score_hood,
            perspective_score_backdoor_left
        ],
    }

  - **Beispielanfrage**:
      ```bash
      curl -X GET -H "Content-Type: application/json" -d '{"data_path": "data/0a1d0d53-eaa4-4f42-9ea7-2197bd183520.jpg}' http://127.0.0.1:5000/predict
      ```

  - **Fehlerbehandlung**:
    - Wenn der Parameter `data_path`  fehlt oder leer ist:
      - Fehlermeldung: `Missing / Empty parameter: data_path`

    - Wenn `data_path` ungültig ist / nicht existiert:
      - Fehlermeldung: `Invalid data_path value. Directory or File does not exist`

### Test
- Führe Tests aus:
     ```bash
    pytest
    ```
  

## Training
The model used in this application was inspired by the following papers:
- [An Evaluation of Pre-trained Models for Feature Extraction in Image Classification](https://arxiv.org/pdf/2310.02037)
- [Backbones-Review: Feature Extraction Networks for Deep Learning and Deep Reinforcement Learning Approaches](https://arxiv.org/pdf/2206.08016)

## Architecture
The architecture includes the following backbones for feature extraction:
- **Backbone for Feature Extraction**: The architecture includes popular backbones for feature extraction, chosen based on their effectiveness in feature extraction tasks.
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

## Metrics
- **MSE (Mean Squared Error)**: Measures the average squared difference between the predicted and actual values.
- **MAE (Mean Absolute Error)**: Measures the average absolute difference between the predicted and actual values.
- **RMSE (Root Mean Squared Error)**: Measures the square root of the average squared difference between the predicted and actual values.
- **R-squared (R²)**: Represents the proportion of the variance in the dependent variable that is predictable from the independent variables.

## Random Grid Search
A small random grid search was conducted to optimize hyperparameters and improve model performance.

## Evaluation

### Quantitative Evaluation

Quantitative evaluation involves assessing the model's performance using the mentioned metrics (MSE, MAE, RMSE, R-squared) on a separate test dataset.

### Qualitative Evaluation

- Qualitative evaluation involves **visually inspecting predictions** on random data samples to understand how well the model performs in practice. This may include visualizing random images alongside their predicted scores.

- Qualitative evaluation involves **checking predictions distribution** can be compared with the distribution of actual labels to assess how well the model captures the underlying patterns in the data.

### Model Selection

After evaluation of the ResNet18, Inception_v3, EfficientNet_b0, and ViT_b_16 based models, the ResNet18 backbone based model appears promising and has been selected for further evaluation and tuning.

The regression model seems to perform reasonably well. The MSE, MAE, and RMSE are relatively low, indicating that the model's predictions are close to the actual values. Additionally, the R-squared value of 0.747 indicates a good fit between predicted and actual values, suggesting that the model captures a significant portion of the variance in the data.

However, it's important to note that we need to be careful of potential issues such as data imbalance or bias during further evaluation and tuning.

## Model Optimization 

### Note

Die vorgeschlagene App dient als Proof of Concept und ist nicht für den Einsatz als produktionsfähiges Modell gedacht.
Weitere Verfeinerungen und Optimierungen sind für den Einsatz in realen Anwendungen erforderlich.

<br> 

The proposed app serves as a proof of concept and is not intended to be a production-ready model.
Further refinement and optimization are necessary for deployment in real-world applications.

### Model Optimization Suggestions (To do)
1. **Hyperparameter Tuning**: Conduct a more extensive hyperparameter search to fine-tune the model's performance.
2. **Data Augmentation**: Rigorous and careful training data Augmentation to improve model generalization.
3. **Transfer Learning**: Explore pre-training domain-specific models for fine-tuning.
4. **Regularization Techniques**: Implement regularization techniques such as dropout or weight decay to prevent overfitting.
5. **Model Compression**: Investigate techniques for model compression to reduce the model size and computational overhead.
6. **Hardware Acceleration**: Utilize hardware accelerators such as GPUs or TPUs to speed up model training and inference.


## Logging

The application logs important events to a file named `api.log` located in the project directory. The log file captures requests received, validation errors, and predictions made.
E
