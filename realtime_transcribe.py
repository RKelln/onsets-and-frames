#!/usr/bin/env python3
"""
Realtime audio piano audio transcription.

You need Python 3.7 or newer to run this.

"""

# https://magic.io/blog/uvloop-blazing-fast-python-networking/

import argparse
import asyncio
import math
import os
import queue
import re
import shutil
import sys
import time

from aioconsole import ainput
from mir_eval.util import midi_to_hz
from mir_eval.util import hz_to_midi
import numpy as np
from numpy_ringbuffer import RingBuffer
import sounddevice as sd
import soundfile
import mido
import rtmidi

from onsets_and_frames import *

usage_line = ' press <enter> to quit, +<enter> or -<enter> to change scaling '

DEFAULT_FREQ_RANGE = [100, 2000]


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

try:
    columns, _ = shutil.get_terminal_size()
except AttributeError:
    columns = 80


# Create a nice output gradient using ANSI escape sequences.
# Stolen from https://gist.github.com/maurisvh/df919538bcef391bc89f
colors = 30, 34, 35, 91, 93, 97
chars = ' :%#\t#%:'
gradient = []
for bg, fg in zip(colors, colors[1:]):
    for char in chars:
        if char == '\t':
            bg, fg = fg, bg
        else:
            gradient.append('\x1b[{};{}m{}'.format(fg, bg + 10, char))

class Output:
    def __init__(self):
        pass

    def send(self, messages):
        if len(messages) > 0:
            print(messages)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass


class MidiOutput(Output):
    # NOTE: midi channels indexed from 1-16 but mido uses 0-15
    def __init__(self, midi_port, midi_channel):
        self.port_name = midi_port
        self.channel = min(15, max(0, midi_channel - 1))

    def send(self, messages):
        for m in messages:
            m.channel = self.channel
            print(self.port_name, m)
            self.port.send(m)

    def close(self):
        self.port.close()

    def __enter__(self):
        self.port = mido.open_output(self.port_name)
        return self

    def __exit__(self, type, value, traceback):
        self.port.close()


async def inputstream_generator(channels=1, **kwargs):
    """Generator that yields blocks of input data as NumPy arrays."""
    q_in = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def callback(indata, frame_count, time_info, status):
        loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))

    stream = sd.InputStream(callback=callback, channels=channels, samplerate=SAMPLE_RATE, dtype='int16', **kwargs)
    with stream:
        while True:
            indata, status = await q_in.get()
            yield indata, status


async def ascii_spectrogram(fftsize, gain, low_bin, columns, **kwargs):
    """Show ASCII spectrogram of the audio input."""
    async for indata, status in inputstream_generator(**kwargs):
        if status:
            text = ' ' + str(status) + ' '
            print('\x1b[34;40m', text.center(columns, '#'),
                  '\x1b[0m', sep='')
        if any(indata):
            magnitude = np.abs(np.fft.rfft(indata[:, 0], n=fftsize))
            magnitude *= gain / fftsize
            line = (gradient[int(np.clip(x, 0, 1) * (len(gradient) - 1))]
                    for x in magnitude[low_bin:low_bin + columns])
            print(*line, sep='', end='\x1b[0m\n')
        else:
            print('no input')


async def transcribe_frame(model, onset_threshold, frame_threshold, device, output, **kwargs):

    sequence_length = 327680 // 100  # // 10 = 10ms,  // 100 = 5ms
    now = time.perf_counter()
    buf = RingBuffer(capacity=sequence_length, dtype= 'int16')
    count = 0

    async for indata, status in inputstream_generator(**kwargs):
        if status:
            text = ' ' + str(status) + ' '
            print('\x1b[34;40m', text.center(columns, '#'),
                  '\x1b[0m', sep='')
        if any(indata):
            buf.extend(indata[:, 0])
            count += len(indata)
            if count >= sequence_length:
                now = time.perf_counter()
                count = 0
                audio = torch.ShortTensor(np.array(buf)).to(device)
                audio = audio.float().div_(32768.0)

                predictions = transcribe(model, audio)
                messages = to_midi(predictions, onset_threshold, frame_threshold)
                if len(messages) > 0:
                    output.send(messages)

                #print("transcribe duration:", 1000 * (time.perf_counter() - now), 'ms')
        else:
            count += 1
            if count % 1000 == 0:
                print('no input')

async def wait_for_input():
    response = await ainput('')
    if response in ('', 'q', 'Q'):
        return None, False
    result = {'gain': 1.0}
    for ch in response:
        if ch == '+':
            result['gain'] *= 2.
        elif ch == '-':
            result['gain'] *= 0.5
        else:
            print('\x1b[31;40m', usage_line.center(args.columns, '#'),
                  '\x1b[0m', sep='')
            return None, True
    return result, True


async def wait_first(*futures):
    ''' Return the result of the first future to finish. Cancel the remaining futures. 
    From https://stackoverflow.com/questions/31900244/select-first-result-from-two-coroutines-in-asyncio 
    '''
    done, pending = await asyncio.wait(futures,
        return_when=asyncio.FIRST_COMPLETED)
    gather = asyncio.gather(*pending)
    gather.cancel()
    try:
        await gather
    except asyncio.CancelledError:
        pass
    return done.pop().result()


async def main(list_devices=None, audio_device=None, 
    freq_range=DEFAULT_FREQ_RANGE, 
    block_duration=10,
    columns=80,
    gain=10,
    model_file = None,
    ml_device = 'cpu',
    midi_port = None,
    midi_channel = 1,
    verbose = False,
    **kwargs):

    if list_devices:
        print("Audio input available:")
        print(sd.query_devices())
        print("\nMidi output available:")
        print(mido.get_output_names())
        parser.exit(0)

    if model_file is None:
        print("Must supply the name of the model file")
        parser.exit(1)

    if not os.path.exists(model_file):
        print("Cannot find model file:", model_file)
        parser.exit(1)

    audio_input_info = sd.query_devices(audio_device, 'input')

    if verbose:
        print("Audio input:")
        print(sd.get_portaudio_version())
        print(audio_input_info)
        print("Midi output:")
        print(midi_port, "channel:", midi_channel)
        print("Model file:", model_file)

    # construct output 
    if midi_port is None:
        output_handler = Output()
    else:
        # check if port is in format: port:channel
        r = re.compile(r'(\d+):(\d+)')
        match = r.search(midi_port)
        if match:
            midi_port = match.group(1) # group(0) is entire match
            midi_channel = int(match.group(2))
        # find midi port match by name
        midi_port = next((x for x in mido.get_output_names() if r.search(x)), midi_port)
        print("Output to midi port:", midi_port, "channel:", midi_channel)
        output_handler = MidiOutput(midi_port, midi_channel)

    with torch.no_grad():
        model = torch.load(model_file, map_location=ml_device).eval()

        with output_handler:
            audio_task = asyncio.create_task(
                transcribe_frame(
                    model=model, 
                    onset_threshold=kwargs['onset_threshold'], 
                    frame_threshold=kwargs['frame_threshold'],
                    device=ml_device,
                    output=output_handler))
            input_task = asyncio.create_task(wait_for_input())

            print("Listening on", audio_input_info['name'] ,"...")

            try:
                result, ok = await wait_first(audio_task, input_task)
                if ok == False:
                    sys.exit()
                for key, value in result.items():
                    if key == 'gain':
                        gain *= value
            except asyncio.CancelledError:
                print('\nListening cancelled')
                return



def transcribe(model, audio):

    mel = melspectrogram(audio.reshape(-1, audio.shape[-1])[:, :-1]).transpose(-1, -2)
    onset_pred, offset_pred, _, frame_pred, velocity_pred = model(mel)

    predictions = {
            'onset': onset_pred.reshape((onset_pred.shape[1], onset_pred.shape[2])),
            'offset': offset_pred.reshape((offset_pred.shape[1], offset_pred.shape[2])),
            'frame': frame_pred.reshape((frame_pred.shape[1], frame_pred.shape[2])),
            'velocity': velocity_pred.reshape((velocity_pred.shape[1], velocity_pred.shape[2]))
        }

    return predictions


def transcribe_file(model_file, flac_paths, save_path, sequence_length,
                  onset_threshold, frame_threshold, device):
    
    model = torch.load(model_file, map_location=device).eval()
    summary(model)

    for flac_path in flac_paths:
        print(f'Processing {flac_path}...', file=sys.stderr)
        audio = load_and_process_audio(flac_path, sequence_length, device)
        predictions = transcribe(model, audio)

        p_est, i_est, v_est = extract_notes(predictions['onset'], predictions['frame'], predictions['velocity'], onset_threshold, frame_threshold)

        scaling = HOP_LENGTH / SAMPLE_RATE

        i_est = (i_est * scaling).reshape(-1, 2)
        p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

        os.makedirs(save_path, exist_ok=True)
        pred_path = os.path.join(save_path, os.path.basename(flac_path) + '.pred.png')
        save_pianoroll(pred_path, predictions['onset'], predictions['frame'])
        midi_path = os.path.join(save_path, os.path.basename(flac_path) + '.pred.mid')
        save_midi(midi_path, p_est, i_est, v_est)


def to_midi(predictions, onset_threshold, frame_threshold):
    """
    pitches: np.ndarray of bin_indices
    intervals: list of (onset_index, offset_index)
    velocities: list of velocity values
    """
    pitches, intervals, velocities = extract_notes(predictions['onset'], predictions['frame'], predictions['velocity'], onset_threshold, frame_threshold)

    scaling = HOP_LENGTH / SAMPLE_RATE

    intervals = (intervals * scaling).reshape(-1, 2)
    pitches = np.array([midi_to_hz(MIN_MIDI + midi) for midi in pitches])

    #ticks_per_second = file.ticks_per_beat * 2.0
    ticks_per_second = 120

    events = []
    for i in range(len(pitches)):
        events.append(dict(type='on', pitch=pitches[i], time=intervals[i][0], velocity=velocities[i]))
        events.append(dict(type='off', pitch=pitches[i], time=intervals[i][1], velocity=velocities[i]))
    events.sort(key=lambda row: row['time'])

    midi_messages = []
    for event in events:
        velocity = int(event['velocity'] * 127)
        if velocity > 127:
            velocity = 127
        pitch = int(round(hz_to_midi(event['pitch'])))
        midi_messages.append(mido.Message('note_' + event['type'], note=pitch, velocity=velocity))

    return midi_messages


def load_and_process_audio(flac_path, sequence_length, device):

    audio, sr = soundfile.read(flac_path, dtype='int16')
    assert sr == SAMPLE_RATE

    audio = torch.ShortTensor(audio)

    if sequence_length is not None:
        audio_length = len(audio)
        begin = 0
        end = min(audio_length, begin + sequence_length)

        audio = audio[begin:end].to(device)
    else:
        audio = audio.to(device)

    audio = audio.float().div_(32768.0)

    return audio





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=usage_line)
    
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose')

    # model args
    parser.add_argument('model_file', nargs='?', type=str, default=None)
    parser.add_argument('--save-path', type=str, default='.')
    parser.add_argument('--sequence-length', default=None, type=int)
    parser.add_argument('--onset-threshold', default=0.5, type=float)
    parser.add_argument('--frame-threshold', default=0.5, type=float)
    parser.add_argument('--ml-device', dest='ml_device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint', default=None, type=int)

    # spectrogram args
    parser.add_argument('-l', '--list-devices', action='store_true',
                        help='list audio devices and exit')
    parser.add_argument('-b', '--block-duration', type=float,
                        metavar='DURATION', default=50,
                        help='block size (default %(default)s milliseconds)')
    parser.add_argument('-c', '--columns', type=int, default=columns,
                        help='width of spectrogram')
    
    # audio args
    parser.add_argument('-d', '--audio-device', type=int_or_str, dest='audio_device',
                        help='input device (numeric ID or substring)')
    parser.add_argument('-g', '--gain', type=float, default=10,
                        help='initial gain factor (default %(default)s)')
    parser.add_argument('-r', '--range', dest='freq_range', type=float, nargs=2,
                        metavar=('LOW', 'HIGH'), default=DEFAULT_FREQ_RANGE,
                        help='frequency range (default %(default)s Hz)')
    
    # midi output args
    parser.add_argument('-p', '--port', type=str, dest='midi_port',
                        help='midi port (string)')
    parser.add_argument('--channel', type=int, dest='midi_channel', default=1,
                        help='midi channel (default %(default)s)')

    args = parser.parse_args()

    try:
        asyncio.run(main(**vars(args)))
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')



