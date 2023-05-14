import audiomentations

## The benefit of this class is that it is optimized data augmentation for classical music
"""
AddGaussianSNR: Instead of adding Gaussian noise directly, the AddGaussianSNR augmentation technique adds noise with a specific Signal-to-Noise Ratio (SNR). By using a higher SNR range (e.g., 15 to 25 dB), you can introduce subtle noise that preserves the clarity and quality of classical music.
TimeStretch: The time stretch augmentation is adjusted to have a smaller range (e.g., 0.9 to 1.1). This helps maintain the original tempo of classical music while introducing slight variations in duration.
PitchShift: The pitch shift range is reduced (e.g., -2 to 2 semitones) to avoid drastic changes in the tonality of classical music. This allows for minor variations in pitch while preserving the overall character.
Shift: The shift range is decreased (e.g., -0.1 to 0.1) to provide slight temporal shifts without altering the structure of the classical music recording significantly.
"""

class ClassicalAudioDataAugmenter:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.augmenter = self.create_augmenter()

    def create_augmenter(self):
        augmenter = audiomentations.Compose([
            audiomentations.AddGaussianSNR(min_SNR=15, max_SNR=25, p=0.5),
            audiomentations.TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
            audiomentations.PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
            audiomentations.Shift(min_fraction=-0.1, max_fraction=0.1, p=0.5),
        ])
        return augmenter

    def augment_audio(self, audio):
        augmented_audio = self.augmenter(samples=audio, sample_rate=self.sample_rate)
        return augmented_audio