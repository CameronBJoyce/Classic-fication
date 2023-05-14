"""
Cameron Joyce 

 Be sure to move the files you want to use out of their respective
 folders and remove those imports you are not using from this run
 script.
"""

# Import necessary libraries and classes
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from model_evaluator import ModelEvaluator
from cnn import CNNModel
from gru import GRUModel
from vgg import VGGAudioModel
from data_balancer import DataBalancer
from ensemble import  EnsembleModel

# Load the data and labels (replace with your own data loading code)
data = 'data/your_audio_data.wav'  # Audio data
labels = 'data/your_labels.csv'  # Corresponding labels

# Encode labels as integers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data, encoded_labels, test_size=0.2, random_state=42)

# Balance the training data
data_balancer = DataBalancer()
x_train_balanced, y_train_balanced = data_balancer.balance_data(x_train, y_train)

# Create individual models (e.g., CNN, GRU, VGG)
cnn_model = CNNModel(num_classes=len(label_encoder.classes__))
gru_model = GRUModel(num_classes=len(label_encoder.classes__))
vgg_model = VGGAudioModel(num_classes=len(label_encoder.classes__))

# Train individual models
cnn_model.train(x_train_balanced, y_train_balanced)
gru_model.train(x_train_balanced, y_train_balanced)
vgg_model.train(x_train_balanced, y_train_balanced)

# Create an ensemble model with the individual models
ensemble_models = [cnn_model, gru_model, vgg_model]
ensemble_model_names = ['CNN', 'GRU', 'VGG']
ensemble_model = EnsembleModel(ensemble_models, ensemble_model_names)

# Evaluate the models
evaluator = ModelEvaluator(ensemble_models, ensemble_model_names)
evaluation_results = evaluator.evaluate(x_test, y_test)

# Compare the models
evaluator.compare_models(evaluation_results)

# Plot the confusion matrix for the ensemble model
y_pred_ensemble = ensemble_model.predict(x_test)
evaluator.plot_confusion_matrix(y_test, y_pred_ensemble, labels=label_encoder.classes_)

# Plot the ROC curve for the ensemble model
y_pred_probs_ensemble = ensemble_model.predict_proba(x_test)[:, 1]  # Assuming binary classification
evaluator.plot_roc_curve(y_test, y_pred_probs_ensemble)
