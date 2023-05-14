import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc


class ModelEvaluator:
    def __init__(self, models, model_names):
        self.models = models
        self.model_names = model_names

    def evaluate(self, x_test, y_test):
        evaluation_results = {}
        for model, name in zip(self.models, self.model_names):
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            cm = confusion_matrix(y_test, y_pred)
            fpr, tpr, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(fpr, tpr)

            evaluation_results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm,
                'fpr': fpr,
                'tpr': tpr,
                'roc_auc': roc_auc
            }

        return evaluation_results

    def plot_confusion_matrix(self, y_test, y_pred, labels):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    def plot_roc_curve(self, y_test, y_pred_probs):
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()

    def compare_models(self, evaluation_results):
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

        for metric in metrics:
            scores = [evaluation_results[name][metric] for name in self.model_names]
            plt.figure(figsize=(10, 6))
            plt.bar(self.model_names, scores)
            plt.xlabel('Models')
            plt.ylabel(metric.capitalize())
            plt.title(f'{metric.capitalize()} Comparison')
            plt.show()
