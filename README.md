# Classic-fication :musical_score:
## Classical Music Classification with Deep Learning
### by Cameron Joyce

This project focuses on building deep learning models for classifying classical music. I explore different architectures such as Convolutional Neural Networks (CNNs), Gated Recurrent Units (GRUs), and VGG-like models, along with techniques like data augmentation, data balancing, and ensemble modeling.

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

To extract features from the audio recordings, I use the Librosa library. It provides various audio processing functions such as Mel spectrogram, MFCC, and chroma features. These features capture important characteristics of the audio and serve as input for our deep learning models.

## Data Augmentation :level_slider:

To enhance the training dataset, I apply data augmentation techniques to create additional samples. This helps in improving the model's generalization capabilities. Common augmentation techniques for audio data include pitch shifting, time stretching, and adding background noise. Specific changes for classical music include:

- AddGaussianSNR: Instead of adding Gaussian noise directly, the AddGaussianSNR augmentation technique adds noise with a specific Signal-to-Noise Ratio (SNR). By using a higher SNR range (e.g., 15 to 25 dB), you can introduce subtle noise that preserves the clarity and quality of classical music.

- TimeStretch: The time stretch augmentation is adjusted to have a smaller range (e.g., 0.9 to 1.1). This helps maintain the original tempo of classical music while introducing slight variations in duration.

- PitchShift: The pitch shift range is reduced (e.g., -2 to 2 semitones) to avoid drastic changes in the tonality of classical music. This allows for minor variations in pitch while preserving the overall character.

- Shift: The shift range is decreased (e.g., -0.1 to 0.1) to provide slight temporal shifts without altering the structure of the classical music recording significantly.

## Model Architectures :building_construction:

I experiment with different deep learning architectures for classical music classification:

- Convolutional Neural Networks (CNNs): CNNs are effective in capturing local patterns and spatial dependencies in data. I design a CNN model tailored for audio data, considering the input shape and number of classes.

- Gated Recurrent Units (GRUs): GRUs are recurrent neural networks known for their ability to model sequential data. I create a GRU-based model that captures temporal dependencies in the audio features.

- VGG-like Models: Although VGG models were originally designed for image classification, I adapt the concept to audio data. I convert the audio data into spectrograms and use a pre-trained VGG-like architecture to extract features from the spectrograms.

## Data Balancing

To address class imbalance in the dataset, I apply data balancing techniques. Specifically, I oversample the minority classes to match the sample count of the majority class. This ensures that the model is not biased towards the majority class and can learn from the entire dataset effectively.

## Ensemble Models :musical_keyboard: :violin: :trumpet:

I combine multiple models using ensemble techniques to improve classification performance. By averaging the predictions of individual models, I create an ensemble model that benefits from the strengths of each model. The ensemble approach can enhance the overall accuracy and robustness of the classification system.

## Model Evaluation and Comparison

I provide a ModelEvaluator class that enables comprehensive evaluation and comparison of different models. It calculates evaluation metrics such as accuracy, precision, recall, F1-score, confusion matrix, and ROC curve. The class also includes visualization methods for better understanding of the model performance.

## Optimization for Classical Music :control_knobs:

The project takes into consideration the characteristics of classical music to optimize the models. Classical music often contains complex compositions, nuanced dynamics, and a wide range of instruments. The choice of feature extraction methods, model architectures, and data augmentation techniques are tailored to capture these characteristics and improve the classification accuracy specifically for classical music.

Feel free to adapt the code and experiment with different models, feature extraction techniques, or data augmentation methods to further optimize the classification of classical music.

For detailed implementation, refer to the code files provided in this repository.