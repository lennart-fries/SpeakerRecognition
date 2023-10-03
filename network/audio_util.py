import random
import torch
import torchaudio
from torchaudio import transforms


class AudioUtil:
    @staticmethod
    def open(audio_file):
        '''
        Loads an audiofile using the torchaudio framework
            Parameters:
                 audio_file (str): Specifies the path to an audio file
            Returns:
                (sig, sr): A tuple consisting of the signal and sample rate of the file
        '''
        sig, sr = torchaudio.load(audio_file)

        return (sig, sr)

    @staticmethod
    def rechannel(aud, new_channel: int):
        '''
        Converts a given signal to another channel
            Parameters:
                 aud (tuple): A tuple containing both a audio signal and sample rate
                 new_channel (int): An integer corresponding to the channel (1: Mono / 2:Stereo)
            Returns:
                (resig, sr): A tuple consisting of the rechanneled (Mono / Stereo) signal and its sample rate
        '''
        sig, sr = aud

        if sig.shape[0] == new_channel:
            return aud

        if new_channel == 1:
            resig = sig[:1, :]
        else:
            resig = torch.cat([sig, sig])

        return ((resig, sr))

    @staticmethod
    def resample(aud, new_sr: int):
        '''
        Resamples the given signal to a given sample rate
            Parameters:
                aud (tuple): A tuple containing both a audio signal and sample rate
                new_sr (int): An integer corresponding to the new sampling rate
            Returns:
                (resig, new_sr): A tuple consisting of the resampled signal and its new sampling rate
        '''
        sig, sr = aud

        if sr == new_sr:
            return aud

        num_channels = sig.shape[0]
        resig = torchaudio.transforms.Resample(sr, new_sr)(sig[:1, :])

        if num_channels > 1:
            retwo = torchaudio.transforms.Resample(sr, new_sr)(sig[1:, :])
            resig = torch.cat([resig, retwo])

        return ((resig, new_sr))

    @staticmethod
    def pad_trunc(aud, max_ms: int):
        '''
        Standardizes the length of the signal to a given duration
            Parameters:
                aud (tuple): A tuple containing both a audio signal and sample rate
                max_ms (int): An integer value describing the duration in milliseconds
            Returns:
                (sig, sr): The signal in the specified duration and its sample rate
        '''
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr//1000 * max_ms

        if sig_len > max_len:
            # Truncate to given length
            sig = sig[:, :max_len]
        elif sig_len < max_len:
            # Pad at start and end of signal, define lengths
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with zeroes
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)

        return (sig, sr)

    @staticmethod
    def time_shift(aud, shift_limit: float):
        '''
        Applies a timeshift to the audio signal, values shifting over the edge wrap around
            Parameters:
                aud (tuple): A tuple containing both a audio signal and sample rate
                shift_limit (float): A float number specifying the upper limit for the shift
            Returns:
                (sig, sr): A tuple containing the shifted audio data and its sample rate
        '''
        sig, sr = aud
        _, sig_len = sig.shape

        shift_amt = int(random.random() * shift_limit * sig_len)

        return (sig.roll(shift_amt), sr)

    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        '''
        Transforms given audio data into a mel spectrogram
            Parameters:
                aud (tuple): A tuple containing both a audio signal and sample rate
                n_mels (int): An integer specifying the size of the mel filterbanks
                n_fft (int): An integer specifying the size of the fast-fourier transform
                hop_len: Specifies the length of the hops during transformation, set to None natively
            Returns:
                spec: The transformed spectrogram
        '''
        sig, sr = aud
        top_db = 80

        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)

        return (spec)

    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        '''
        Augments a given mel spectrogram using frequency (horizontal) and time (vertical) masking
            Parameters:
                spec: The spectrogram to be augmented
                max_mask_pct (float): A float value specifying the percentage multiplier for max. mask lengths
                n_freq_masks (int): An integer specifying the maximum possible length of the frequency mask
                n_time_masks (int): An integer specifying the maximum possible length of the time mask
        '''
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec