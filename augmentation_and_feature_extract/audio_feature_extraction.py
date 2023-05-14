import librosa

class AudioFeatureExtractor:
    def __init__(self, sr=22050, n_mfcc=13, hop_length=512):
        self.sr = sr  # Sample rate
        self.n_mfcc = n_mfcc  # Number of MFCC coefficients to extract
        self.hop_length = hop_length  # Hop length for spectrogram computation

    def extract_features(self, audio_path):
        # Load audio file
        y, sr = librosa.load(audio_path, sr=self.sr)

        # Extract features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, hop_length=self.hop_length)
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=self.hop_length)

        return mfcc, spectrogram