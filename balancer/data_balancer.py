import numpy as np
from sklearn.utils import shuffle

class DataBalancer:
    def __init__(self):
        self.class_counts = None

    def calculate_class_counts(self, labels):
        unique_classes, class_counts = np.unique(labels, return_counts=True)
        self.class_counts = dict(zip(unique_classes, class_counts))

    def balance_data(self, features, labels):
        if self.class_counts is None:
            self.calculate_class_counts(labels)

        max_class_count = max(self.class_counts.values())
        balanced_features = []
        balanced_labels = []

        for class_label in self.class_counts.keys():
            class_indices = np.where(labels == class_label)[0]
            class_samples = len(class_indices)
            oversampled_features = features[class_indices]
            oversampled_labels = labels[class_indices]

            if class_samples < max_class_count:
                repeat_times = int(max_class_count / class_samples)
                remaining_samples = max_class_count % class_samples
                oversampled_features = np.repeat(oversampled_features, repeat_times, axis=0)
                oversampled_labels = np.repeat(oversampled_labels, repeat_times, axis=0)

                if remaining_samples > 0:
                    random_indices = np.random.choice(class_samples, remaining_samples, replace=False)
                    oversampled_features = np.concatenate((oversampled_features, features[class_indices[random_indices]]))
                    oversampled_labels = np.concatenate((oversampled_labels, labels[class_indices[random_indices]]))

            balanced_features.append(oversampled_features)
            balanced_labels.append(oversampled_labels)

        balanced_features = np.concatenate(balanced_features)
        balanced_labels = np.concatenate(balanced_labels)
        balanced_features, balanced_labels = shuffle(balanced_features, balanced_labels)

        return balanced_features, balanced_labels