import tensorflow as tf
import numpy as np
import librosa
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

class VGGAudioModel:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)
        return model

    def preprocess_audio(self, audio_path):
        # Load audio file
        audio, sr = librosa.load(audio_path)

        # Convert audio to spectrogram
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

        # Resize spectrogram to match VGG input size
        resized_spectrogram = tf.image.resize(spectrogram, size=(224, 224))
        resized_spectrogram = resized_spectrogram[..., np.newaxis]
        resized_spectrogram = np.repeat(resized_spectrogram, 3, axis=2)

        return resized_spectrogram

    def preprocess_labels(self, labels):
        return to_categorical(labels, num_classes=self.num_classes)

    def train(self, x_train, y_train, batch_size=32, epochs=10, validation_data=None):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=validation_data)

    def evaluate(self, x_test, y_test):
        loss, accuracy = self.model.evaluate(x_test, y_test)
        print("Loss:", loss)
        print("Accuracy:", accuracy)