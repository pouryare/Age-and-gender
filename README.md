# Age and Gender Prediction from chest X-Ray Images

This project implements a deep learning model to predict age and gender from chest X-ray images. It includes a data preprocessing pipeline, model training scripts, and a Flask web application for easy deployment and usage.

## Project Structure

- `Age and gender.ipynb`: Main Jupyter notebook for data preprocessing, model training, and evaluation (to be run in Google Colab)
- `Web App/app.py`: Flask web application for serving predictions
- `Web App/model.py` : Contains functions for image preprocessing and prediction

## Features

- Data loading and preprocessing from Kaggle dataset
- Flexible CNN model architecture for age and gender prediction
- Data augmentation techniques
- Model training with learning rate scheduling
- Evaluation metrics and visualizations
- Flask web application for easy deployment

## Prerequisites

- Google Colab account
- Kaggle account (for dataset access)
- Python 3.7+
- TensorFlow 2.x
- Flask
- OpenCV
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Installation

1. Open the `Age and gender.ipynb` notebook in Google Colab.

2. Ensure you have a Kaggle account and API token set up. If not, follow these steps:
   - Create a Kaggle account at https://www.kaggle.com
   - Go to your account settings and create a new API token
   - Download the `kaggle.json` file

3. Upload the `kaggle.json` file to your Google Drive in the appropriate directory (as specified in the notebook).

## Usage

### Training the Models

1. Open the `Age and gender.ipynb` notebook in Google Colab.

2. Run all cells in the notebook sequentially. This will:
   - Mount your Google Drive
   - Set up the Kaggle API
   - Download the dataset
   - Preprocess the X-ray images
   - Create and train separate models for age and gender prediction
   - Evaluate the models and display performance metrics
   - Save the trained models to your Google Drive

### Running the Web Application

The Flask web application is located in the `Web App` folder. To run it locally:

1. Ensure you have Flask installed:
   ```
   pip install flask
   ```

2. Navigate to the `Web App` folder:
   ```
   cd 'Web App'
   ```

3. Run the Flask app:
   ```
   python app.py
   ```

4. Open a web browser and go to `http://localhost:5000` to use the prediction interface.

## Model Architecture

The project uses a Convolutional Neural Network (CNN) with the following structure:

- Multiple convolutional layers with batch normalization and max pooling
- Global average pooling
- Dense layers with dropout for regularization
- Output layer (sigmoid activation for gender, linear for age)

## Performance

The models are evaluated using the following metrics:

- Gender Prediction: Accuracy, Precision, Recall, F1-score, ROC curve
- Age Prediction: R2 score, Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE)

Detailed performance metrics and visualizations are generated during the evaluation phase in the Colab notebook.

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset provided by Felipe Kitamura on Kaggle
- Inspired by various age and gender prediction projects in the medical imaging domain

