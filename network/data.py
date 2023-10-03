import pandas as pd
import glob
import torch
from pathlib import Path
from tqdm import tqdm
from network.audio_util import AudioUtil
from torch.utils.data import Dataset, DataLoader, random_split


class Data:
    '''
    Prepares the data in the provided path

        Parameters:
            path (str): A string corresponding to the path of the data

        Data needs to be in the form of .wav for noise and .flac for voice data. Can change manually if needed
    '''
    def __init__(self, path: str):
        self.path = path

    def _metadata(self, speaker: bool, voices: int):
        '''
        Prepares a dataframe of the metadata

            Parameters:
                speaker (bool): Describes the datasets goal: speaker recognition (true) or speech recognition (false)
                voices (int): An integer corresponding to the ammount of voices to recognize

            Returns:
                df: A dataframe containing the path identifiers and targets for every audio file
        '''

        df = pd.DataFrame(columns=['path', 'target'])
        root = Path(self.path)
        folders = sorted([j.name.split('/')[-1] for j in root.iterdir()])
        for i, folder in enumerate(folders):
            if i is 0:
                files = glob.glob(self.path + folder + '/**/*.wav')
            else:
                files = glob.glob(self.path + folder + '/**/*.flac')

            for file in files:
                if speaker and i <= voices:
                    tmp_df = pd.DataFrame([[file, i]], columns=['path', 'target'])
                elif not speaker and i > 0:
                    tmp_df = pd.DataFrame([[file, 1]], columns=['path', 'target'])
                elif not speaker and i == 0:
                    tmp_df = pd.DataFrame([[file, 0]], columns=['path', 'target'])
                else:
                    break

                df = df.append(tmp_df, ignore_index=True)

        return df

    def _audiodata(self, metadata, duration: float, sample_rate: int = 16000):
        '''
        Transforms the metadata into a labelled audio signal

            Parameters:
                metadata: A dataframe containing both path and target identifiers for each audio file
                duration (float): A floating number corresponding to the duration of singular fragments
                sample_rate (int): An integer corresponding to the sampling hz rate

            Returns:
                df: A dataframe containing both an augmented spectrogram transformation of the audio data and target
        '''
        df = pd.DataFrame(columns=['audio', 'target'])

        for i in tqdm(range(metadata.shape[0])):
            # Get file
            file = metadata.loc[i, 'path']
            id = metadata.loc[i, 'target']

            # Open file path in metadata as tensor
            audio = AudioUtil.open(file)

            # Preprocess the audio
            reaud = AudioUtil.resample(audio, sample_rate)
            rechan = AudioUtil.rechannel(reaud, 2)
            waveform, _ = rechan

            # Amount of slices to split the file into
            # Depends on file-length, sample rate and desired duration
            frames = int((waveform.shape[1] / sample_rate) // duration)

            # For each frame, slice waveform, process audio signal and append
            for i in range(frames):
                # Slice waveform to start from the current frame onwards
                wave = waveform[:, (i * sample_rate):]
                # Truncate the audio to the desired duration
                # Duration for the method in ms, thus it is transformed here
                dur_aud = AudioUtil.pad_trunc((wave, sample_rate), max_ms=int(duration * 1000))

                # Process the audio signal and get augmented spectrogram
                shift_aud = AudioUtil.time_shift(dur_aud, shift_limit=0.4)
                sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
                aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

                # Append the spectrogram and id to the dataframe to return them later
                tmp_df = pd.DataFrame([[aug_sgram, id]], columns=['audio', 'target'])
                df = df.append(tmp_df, ignore_index=True)

        return df

    def prepare_data(self, speaker: bool, voices: int, duration: float, split: float = 0.2):
        '''
        Takes the parameters of the datasets and prepares data loaders for the torch framework

            Parameters:
                speaker (bool): Describes the datasets goal: speaker recognition (true) or speech recognition (false)
                voices (int): An integer corresponding to the ammount of voices to recognize
                duration (float): A floating number corresponding to the duration of singular fragments
                split (float): A floating number describing the percentage of test data to split from the dataset

            Returns:
                train_loader: A torch dataloader object containing the data used for training
                test_loader: A torch dataloader object containing the data used for testing the trained model
        '''
        meta = self._metadata(speaker, voices)
        audio = self._audiodata(meta, duration)

        # Prepare the Dataset and get values for splitting
        dataset = Dataset(audio)
        num_items = len(dataset)
        num_train = round(num_items * (1 - split))
        num_val = num_items - num_train

        train_data, test_data = random_split(dataset, [num_train, num_val], generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

        return train_loader, test_loader


class Dataset(Dataset):
    '''
    Takes data and provides basic dataset functionality for the purposes of the torch framework

        Parameters:
            data: Corresponds to a Dataframe containing both an audio signal and target
    '''
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio = self.data.loc[idx, 'audio']
        target = self.data.loc[idx, 'target']

        return audio, target
