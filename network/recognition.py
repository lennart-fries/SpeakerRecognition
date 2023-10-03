import sounddevice as sd
import torch
from network.audio_util import AudioUtil
import numpy as np
from network.model import single_inference

class LiveRecognition():
    '''
    Provides a class to recognize either speakers or speech depending on the parameters

        Parameters:
            model: A torch-based neural network to use for the training cycle
            speaker (bool): Describes the datasets goal: speaker recognition (true) or speech recognition (false)
            duration (float): A floating number corresponding to the duration of singular fragments
            sample_rate (int): An integer corresponding to the sampling hz rate
    '''
    def __init__(self, model, speaker: bool, duration: float, sample_rate: int = 44100):
        self.model = model
        self.speaker = speaker
        self.duration = duration
        self.sample_rate = sample_rate

    def _transform(self, audio):
        '''
        Applies a transform pipeline on audio data

            Parameters:
                audio: A torch array containing data of an audio signal

            Returns:
                sgram_tensor: An augmented spectrogram in the form of a torch tensor
        '''
        resampled_aud = AudioUtil.resample(audio, 16000)
        rechanneled_aud = AudioUtil.rechannel(resampled_aud, 2)
        fit_aud = AudioUtil.pad_trunc(rechanneled_aud, self.duration*1000)
        shift_aud = AudioUtil.time_shift(fit_aud, 0.4)
        spectrogram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_spectro = AudioUtil.spectro_augment(spectrogram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
        sgram_tensor = torch.as_tensor(np.array([aug_spectro.numpy()]))

        return sgram_tensor

    def classify_live(self):
        '''
        Listens to live data from the sound device (microphone), transforms the input and with the spectrogram
        predicts live audio with the given network

            Returns:
                output (int): An integer value corresponding to the prediction with the most confidence
        '''
        myrec = sd.rec(int(self.duration * self.sample_rate), samplerate=self.sample_rate, channels=1)

        waveform = torch.from_numpy(myrec)
        waveform = waveform.t()

        sd.wait()

        input_spectro = self._transform((waveform, self.sample_rate))
        output = single_inference(self.model, input_spectro)

        return output

    def loop(self, verbose: bool = True):
        '''
        Loops indefinitely, listening to input and classifying it, providing analysis of the output

            Parameters:
                verbose (bool): Specifies whether additional information / analysis shall be printed
        '''
        print('Loop started, begin talking!')
        while True:
            output = self.classify_live()
            print(f'Model predicts {output} as output')
            if self.speaker:
                if verbose:
                    if output == 0:
                        print('Either nobody or not recognized speaker talking')
                    else:
                        print(f'Recognized speaker {output} talking')
            else:
                if verbose:
                    if output == 0:
                        print('Currently nobody is speaking')
                    else:
                        print('Detected somebody speaking')
