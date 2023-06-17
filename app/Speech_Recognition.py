from models.signal import Signal
from core.speech import HMMClassifier


if __name__ == "__main__":

    signal = Signal('audio\\F_voice\\Znormalizowane\\aba_1.wav')
    hmm = HMMClassifier()
    hmm.fit(signal, 'test')
    print(hmm.predict(signal))
