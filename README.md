# Classic-fication :musical_score:
## Classical Music Classification with Deep Learning
### by Cameron Joyce

This project focuses on building deep learning models for classifying classical music. We explore different architectures such as Convolutional Neural Networks (CNNs), Gated Recurrent Units (GRUs), and VGG-like models, along with techniques like data augmentation, data balancing, and ensemble modeling.

## Requirements

- Python 3.6+
- NumPy
- scikit-learn
- TensorFlow
- Keras
- Librosa (for audio processing)
- Matplotlib
- Seaborn

## Data

The project assumes you have a dataset of classical music audio recordings. The dataset should be organized such that each audio file is associated with a corresponding label or class.

## Feature Extraction

To extract features from the audio recordings, we use the Librosa library. It provides various audio processing functions such as Mel spectrogram, MFCC, and chroma features. These features capture important characteristics of the audio and serve as input for our deep learning models.

## Data Augmentation

To enhance the training dataset, we apply data augmentation techniques to create additional samples. This helps in improving the model's generalization capabilities. Common augmentation techniques for audio data include pitch shifting, time stretching, and adding background noise.

## Model Architectures

We experiment with different deep learning architectures for classical music classification:

- Convolutional Neural Networks (CNNs): CNNs are effective in capturing local patterns and spatial dependencies in data. We design a CNN model tailored for audio data, considering the input shape and number of classes.

- Gated Recurrent Units (GRUs): GRUs are recurrent neural networks known for their ability to model sequential data. We create a GRU-based model that captures temporal dependencies in the audio features.

- VGG-like Models: Although VGG models were originally designed for image classification, we adapt the concept to audio data. We convert the audio data into spectrograms and use a pre-trained VGG-like architecture to extract features from the spectrograms.

## Data Balancing

To address class imbalance in the dataset, we apply data balancing techniques. Specifically, we oversample the minority classes to match the sample count of the majority class. This ensures that the model is not biased towards the majority class and can learn from the entire dataset effectively.

## Ensemble Models

We combine multiple models using ensemble techniques to improve classification performance. By averaging the predictions of individual models, we create an ensemble model that benefits from the strengths of each model. The ensemble approach can enhance the overall accuracy and robustness of the classification system.

## Model Evaluation and Comparison

We provide a ModelEvaluator class that enables comprehensive evaluation and comparison of different models. It calculates evaluation metrics such as accuracy, precision, recall, F1-score, confusion matrix, and ROC curve. The class also includes visualization methods for better understanding of the model performance.

## Optimization for Classical Music

The project takes into consideration the characteristics of classical music to optimize the models. Classical music often contains complex compositions, nuanced dynamics, and a wide range of instruments. The choice of feature extraction methods, model architectures, and data augmentation techniques are tailored to capture these characteristics and improve the classification accuracy specifically for classical music.

Feel free to adapt the code and experiment with different models, feature extraction techniques, or data augmentation methods to further optimize the classification of classical music.

For detailed implementation, refer to the code files provided in this repository.

