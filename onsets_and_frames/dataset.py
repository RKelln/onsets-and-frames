from abc import abstractmethod
from glob import glob
import json
import os
import random
import subprocess
import tempfile
import gc

import numpy as np
import soundfile
from torch.utils.data import Dataset
from tqdm import tqdm

from .constants import *
from .midi import parse_midi


class PianoRollAudioDataset(Dataset):
    def __init__(self, path, groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE, preload=True):
        self.path = os.path.abspath(os.path.expanduser(path))
        self.groups = groups if groups is not None else self.available_groups()
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)
        self.preload = preload

        self.data = []
        print(f"Loading {len(groups)} group{'s' if len(groups) > 1 else ''} "
              f"of {self.__class__.__name__} at {path}")

        for group in groups:
            for input_files in tqdm(self.files(group), desc='Loading group %s' % group):
                self.data.append(self.load(*input_files, preload=preload))

    def __getitem__(self, index):
        data = self.data[index]
        result = dict(path=data['path'])

        audio = data['audio']
        if audio is None:
            audio = self.load_audio(data['path'])

        if self.sequence_length is not None:
            audio_length = len(audio)
            step_begin = self.random.randint(audio_length - self.sequence_length) // HOP_LENGTH
            n_steps = self.sequence_length // HOP_LENGTH
            step_end = step_begin + n_steps

            begin = step_begin * HOP_LENGTH
            end = begin + self.sequence_length

            result['audio'] = audio[begin:end]
            result['label'] = data['label'][step_begin:step_end, :]
            result['velocity'] = data['velocity'][step_begin:step_end, :]
        else:
            result['audio'] = audio
            result['label'] = data['label']
            result['velocity'] = data['velocity']

        result['audio'] = result['audio'].float().div_(32768.0)
        result['onset'] = (result['label'] == 3).float()
        result['offset'] = (result['label'] == 1).float()
        result['frame'] = (result['label'] > 1).float()
        result['velocity'] = result['velocity'].float().div_(128.0)

        return result

    def __len__(self):
        return len(self.data)

    @classmethod
    @abstractmethod
    def available_groups(cls):
        """return the names of all available groups"""
        raise NotImplementedError

    @abstractmethod
    def files(self, group):
        """return the list of input files (audio_filename, tsv_filename) for this group"""
        raise NotImplementedError

    def load_audio(self, audio_path):
        audio, sr = soundfile.read(os.path.join(self.path, audio_path), dtype='int16')
        assert sr == SAMPLE_RATE
        return torch.ShortTensor(audio)

    def load(self, audio_path, tsv_path, preload=True):
        """
        load an audio track and the corresponding labels

        Returns
        -------
            A dictionary containing the following data:

            path: str
                the path to the audio file

            audio: torch.ShortTensor, shape = [num_samples] || None
                the raw waveform, or None if preload is False

            label: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains the onset/offset/frame labels encoded as:
                3 = onset, 2 = frames after onset, 1 = offset, 0 = all else

            velocity: torch.ByteTensor, shape = [num_steps, midi_bins]
                a matrix that contains MIDI velocity values at the frame locations
        """

        saved_data_path = os.path.join(self.path, audio_path).replace('.flac', '.pt').replace('.wav', '.pt')

        if os.path.exists(saved_data_path):
            data = torch.load(saved_data_path)
            if not preload:
                data['audio'] = None # remove the audio data
            return data

        audio = self.load_audio(audio_path)
        audio_length = len(audio)

        n_keys = MAX_MIDI - MIN_MIDI + 1
        n_steps = (audio_length - 1) // HOP_LENGTH + 1

        label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
        velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

        tsv_path = os.path.join(self.path, tsv_path)
        midi = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)

        for onset, offset, note, vel in midi:
            left = int(round(onset * SAMPLE_RATE / HOP_LENGTH))
            onset_right = min(n_steps, left + HOPS_IN_ONSET)
            frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
            frame_right = min(n_steps, frame_right)
            offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)

            f = int(note) - MIN_MIDI
            if f < 88 and f >= 0:
                label[left:onset_right, f] = 3
                label[onset_right:frame_right, f] = 2
                label[frame_right:offset_right, f] = 1
                velocity[left:frame_right, f] = vel

        data = dict(path=audio_path, audio=audio, label=label, velocity=velocity)
        torch.save(data, saved_data_path)
        if not preload:
            data['audio'] = None
        return data


class NoisyAudioDataset(PianoRollAudioDataset):

    def __init__(self, path, groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE, preload=True, 
        min_noise_vol=0., max_noise_vol=0.05):
        self.min_noise_vol = min_noise_vol
        self.max_noise_vol = max_noise_vol
        super().__init__(path, groups if groups is not None else ['train'], sequence_length=sequence_length, seed=seed, device=device, preload=preload)

    def load_audio(self, audio_path):
        full_audio_path = os.path.join(self.path, audio_path)

        noise_vol = random.uniform(self.min_noise_vol, self.max_noise_vol)

        # add noise:
        with tempfile.NamedTemporaryFile(suffix='.wav') as temp_input_with_noise:
            self.add_noise(full_audio_path, temp_input_with_noise, 0.1, 'pinknoise')

            audio, sr = soundfile.read(temp_input_with_noise.name, dtype='int16')
            assert sr == SAMPLE_RATE
        
        return torch.ShortTensor(audio)


    def add_noise(self, input_filename, output_filename, noise_vol, noise_type):
        """Add noise to a wav file using sox.

        Args:
            input_filename: Path to the original wav file.
            output_filename: Path to the output wav file that will consist of the input
                file plus noise.
            noise_vol: The volume of the noise to add.
            noise_type: One of "whitenoise", "pinknoise", "brownnoise".

        Raises:
            ValueError: If `noise_type` is not one of "whitenoise", "pinknoise", or
                "brownnoise".
        """
        if noise_type not in ('whitenoise', 'pinknoise', 'brownnoise'):
            raise ValueError('invalid noise type: %s' % noise_type)

        noise_cmd = ['sox', input_filename, '-p', 'synth', noise_type, 'vol', str(noise_vol)]
        mixer_cmd = ['sox', '-m', input_filename, '-', output_filename.name]
        #print(f"Executing: {' '.join(noise_cmd)} | {' '.join(mixer_cmd)}" );

        # subprocess may need double the memory, or large swap space: see README
        # forking causes memory allocation issues, because it copies all the existing python memory
        gc.collect() # tries to reduce memory

        #p = subprocess.run(args, capture_output=True, shell=True, timeout=20.0)

        p1 = subprocess.Popen(noise_cmd, stdout=subprocess.PIPE)
        p2 = subprocess.Popen(mixer_cmd, stdin=p1.stdout, stdout=subprocess.PIPE)
        p1.stdout.close()
        output = p2.communicate()[0]

        # process_handle = subprocess.Popen(
        #   command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        # process_handle.communicate()


class MAESTRO(NoisyAudioDataset):

    def __init__(self, path='data/MAESTRO', groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE, preload=True, **kwargs):
        super().__init__(path, groups if groups is not None else ['train'], sequence_length=sequence_length, seed=seed, device=device, preload=preload, **kwargs)

    @classmethod
    def available_groups(cls):
        return ['train', 'validation', 'test']

    def files(self, group):
        if group not in self.available_groups():
            # year-based grouping
            flacs = sorted(glob(os.path.join(self.path, group, '*.flac')))
            if len(flacs) == 0:
                flacs = sorted(glob(os.path.join(self.path, group, '*.wav')))

            midis = sorted(glob(os.path.join(self.path, group, '*.midi')))
            files = list(zip(flacs, midis))
            if len(files) == 0:
                raise RuntimeError(f'Group {group} is empty')
        else:
            metadata = json.load(open(os.path.join(self.path, MAESTRO_JSON)))
            files = sorted([(os.path.join(self.path, row['audio_filename'].replace('.wav', '.flac')),
                             os.path.join(self.path, row['midi_filename'])) for row in metadata if row['split'] == group])

            files = [(audio if os.path.exists(audio) else audio.replace('.flac', '.wav'), midi) for audio, midi in files]

        result = []
        for audio_path, midi_path in files:
            tsv_filename = midi_path.replace('.midi', '.tsv').replace('.mid', '.tsv')
            if not os.path.exists(tsv_filename):
                midi = parse_midi(midi_path)
                np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')
            result.append(
                (os.path.relpath(audio_path, self.path),
                 os.path.relpath(tsv_filename, self.path))
            )
        return result


class MAPS(PianoRollAudioDataset):
    def __init__(self, path='data/MAPS', groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE):
        super().__init__(path, groups if groups is not None else ['ENSTDkAm', 'ENSTDkCl'], sequence_length, seed, device)

    @classmethod
    def available_groups(cls):
        return ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'ENSTDkAm', 'ENSTDkCl', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']

    def files(self, group):
        flacs = glob(os.path.join(self.path, 'flac', '*_%s.flac' % group))
        tsvs = [f.replace('/flac/', '/tsv/matched/').replace('.flac', '.tsv') for f in flacs]

        assert(all(os.path.isfile(flac) for flac in flacs))
        assert(all(os.path.isfile(tsv) for tsv in tsvs))

        return sorted(zip(flacs, tsvs))


class RandomDataset(NoisyAudioDataset):

    def __init__(self, path='data/random', groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE, preload=True, **kwargs):
        super().__init__(path, groups if groups is not None else ['train'], sequence_length=sequence_length, seed=seed, device=device, preload=preload, **kwargs)

    @classmethod
    def available_groups(cls):
        return ['train', 'validation', 'test']

    def files(self, group):
        # sub directory based grouping
        flacs = sorted(glob(os.path.join(self.path, group, '*.flac')))
        midis = sorted(glob(os.path.join(self.path, group, '*.mid')))
        files = list(zip(flacs, midis))
        if len(files) == 0:
            raise RuntimeError(f'Group {group} is empty')

        result = []
        for audio_path, midi_path in files:
            tsv_filename = midi_path.replace('.mid', '.tsv')
            if not os.path.exists(tsv_filename):
                midi = parse_midi(midi_path)
                np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')
            result.append(
                (os.path.relpath(audio_path, self.path),
                 os.path.relpath(tsv_filename, self.path))
            )
        return result