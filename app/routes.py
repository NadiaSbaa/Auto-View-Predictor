import logging
from flask import request, jsonify
from app import app
from .request_utils import validate_input_data
from perspectives_score_predictor.predict import predict_scores
from utils.model import get_best_model

best_model = get_best_model()

# Configure logging
logging.basicConfig(filename='api.log',
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')


@app.route('/predict', methods=['GET'])
def predict():
    """
    Endpoint for predicting perspective scores on input data.

    Retrieves the data path from the request, validates it, makes predictions, and returns the predicted scores.

    Returns:
        JSON: Predicted scores or error message.

    Raises:
        None
    """
    data_path = request.get_json().get('data_path')

    # Log request
    logging.info(f"Request received - Data_path: {data_path}")

    # Validate input data
    is_valid, error_msg = validate_input_data(data_path)
    if not is_valid:
        # Log validation error
        logging.error(f"Validation error: {error_msg}")
        return jsonify({'error': error_msg}), 400

    # Predict text
    predicted_scores = predict_scores(best_model, data_path)

    # Log prediction
    logging.info(f"Prediction made - Predicted scores: {predicted_scores}")

    return jsonify({'Predicted scores': predicted_scores})


# Run the Flask app if executed directly
if __name__ == '__main__':
    app.run(debug=True)
