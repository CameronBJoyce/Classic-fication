# Cameron Joyce
import numpy as np
from sklearn.metrics import accuracy_score

"""
If you want to form a little symphony try this out:
# create individual models 
cnn_model = CNNModel(num_classes, input_shape)
gru_model = GRUModel(num_classes, input_shape)
vgg_model = VGGAudioModel(num_classes)

# Train individual models
cnn_model.train(x_train, y_train)
gru_model.train(x_train, y_train)
vgg_model.train(x_train, y_train)

# Create an ensemble model with the individual models
ensemble_models = [cnn_model, gru_model, vgg_model]
ensemble_model = EnsembleModel(ensemble_models)

# evaluate the ensemble model
ensemble_model.evaluate(x_test, y_test)
"""


class EnsembleModel:
    def __init__(self, models):
        self.models = models

    def train(self, x_train, y_train):
        for model in self.models:
            model.train(x_train, y_train)

    def predict(self, x_test):
        predictions = []
        for model in self.models:
            pred = model.predict(x_test)
            predictions.append(pred)
        ensemble_predictions = np.mean(predictions, axis=0)
        return np.argmax(ensemble_predictions, axis=1)

    def evaluate(self, x_test, y_test):
        y_pred = self.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)



