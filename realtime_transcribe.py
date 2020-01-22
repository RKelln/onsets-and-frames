#!/usr/bin/env python3
"""
Realtime audio piano audio transcription.

You need Python 3.7 or newer to run this.

List audio devices:

    $ python realtime_transcribe.py -l

Example:

    $ python realtime_transcribe.py -d 7 -p 129:0 models/uni/model-1000000.pt
"""

import argparse
import asyncio
import math
import os
import queue
import re
import shutil
import sys
import time

#from aioconsole import ainput
from mir_eval.util import midi_to_hz
from mir_eval.util import hz_to_midi
import numpy as np
import sounddevice as sd
import soundfile
import mido
import rtmidi

from onsets_and_frames import *

usage_line = ' press <CTRL-C> to quit '

# Note: WINDOW_LENGTH = 2048, but creates a lot of latency,
# smaller than 1024 doesn't have the frequency resolution for mel
# 1024 seems the sweet spot
WINDOW_LENGTH = 1024

PITCHES = MAX_MIDI - MIN_MIDI

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


class Output:
    def __init__(self):
        self.start = time.monotonic()

    async def send(self, messages):
        ms_since_start = int(1000 * (time.monotonic() - self.start))
        for m in messages:
            print('{:8d}> {}'.format(ms_since_start, m))

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass


class MidiOutput(Output):
    # NOTE: midi channels indexed from 1-16 but mido uses 0-15
    def __init__(self, midi_port, midi_channel, save_to=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.port_name = midi_port
        self.channel = min(15, max(0, midi_channel - 1))
        self.midi_file = save_to
        if save_to:
            # create midi file to save to, if exists
            # TODO:
            self.saved_midi = []

    async def send(self, messages):
        ms_since_start = int(1000 * (time.monotonic() - self.start))
        for m in messages:
            m.channel = self.channel
            self.port.send(m)
            if self.verbose:
                print('{:8d}: {:<12}> {}'.format(ms_since_start, self.port_name[:12], m))
        if self.midi_file:
            self.saved_midi.extend(messages)

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
        loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), frame_count, status))

    stream = sd.InputStream(callback=callback, channels=channels, samplerate=SAMPLE_RATE, dtype='int16', **kwargs)
    with stream:
        while True:
            indata, frame_count, status = await q_in.get()
            yield indata, frame_count, status


async def transcribe_frame(model, window, onset_threshold, frame_threshold, device, output, verbose=False, **kwargs):

    last_update = time.monotonic()
    buffer_len = WINDOW_LENGTH
    buf = np.zeros(WINDOW_LENGTH, dtype='int16')
    temp_buf = np.zeros(WINDOW_LENGTH, dtype='int16')
    buf_window_start = buffer_len - window
    buf_end = buf_window_start
    count = 0
    if verbose:
        print(f"Window size: {window} {100 * window / buffer_len}% of buffer")

    # Buffer looks like this:
    # /------------- buffer_len ----------------\
    #                              /-- window --\
    # [  previously predicted data |  new input ]
    #
    # Each window frames of data we predict and roll the data one window.
    # TODO: We then need to remove re-predicted messages.

    melspectrogram = MelSpectrogram(N_MELS, SAMPLE_RATE, buffer_len, HOP_LENGTH, mel_fmin=MEL_FMIN, mel_fmax=MEL_FMAX)
    melspectrogram.to(device)

    transformer = MidiTransformer(onset_threshold, frame_threshold)

    # update frequency information when verbose = True
    update_report_freq = 100
    update_durations = [0 for _ in range(update_report_freq)]
    update_count = 0

    async for indata, frame_count, status in inputstream_generator(**kwargs):
        if status:
            text = ' ' + str(status) + ' '
            print('\x1b[34;40m', text.center(columns, '#'),
                  '\x1b[0m', sep='')

        data = indata[:, 0]
        in_len = len(data)
        if any(data):
            count += frame_count
            #print(count, frame_count, indata.shape) #, buf.shape, buf.maxlen)

            if count < window:
                # append into buffer
                # TODO: check for overruns
                buf[buf_end:(buf_end+in_len)] = data
                buf_end += in_len
                #print("copy into buffer", buf_end, buf)

            # FIXME: We place the buffer at the start of the window, and it can be smaller,
            #        so that the latency is reduced
            else:
                # fill up buffer
                end = buffer_len - buf_end
                buf[buf_end:buffer_len] = data[:end]
                #print("fill buffer: ", buf_end, end, buffer_len)

                # note: buf is copied
                audio = torch.ShortTensor(buf).to(device)
                audio = audio.float().div_(32768.0)

                # predict and convert to midi
                # approx: 5ms to predict
                predictions = transcribe(model, audio, melspectrogram)
                # approx: 2.5ms to extract notes
                midi_messages = transformer.extract_notes(predictions)
                if len(midi_messages) > 0:
                    # approx: < 0.1ms to send
                    await output.send(midi_messages)

                # reset buffer
                count = in_len - end
                buf_end = buf_window_start + count
                if window < buffer_len:
                    #buf = np.roll(buf, -window)
                    # faster than roll (but more memory)
                    temp_buf[:-window] = buf[window:]
                    buf = temp_buf

                buf[buf_window_start:buf_end] = data[end:]
                #print("remaining: ", count)
                #print("buf", buf.shape, buf)

                # track updates
                if verbose:
                    now = time.monotonic()
                    update_durations[update_count] = now - last_update
                    last_update = now
                    update_count += 1
                    if update_count >= update_report_freq:
                        update_count = 0
                        avg_update = sum(update_durations) / len(update_durations)
                        print(int(1000 * avg_update), "ms between updates")

# async def wait_for_input():
#     response = await ainput('')
#     if response in ('', 'q', 'Q'):
#         return None, False
#     result = {'gain': 1.0}
#     for ch in response:
#         if ch == '+':
#             result['gain'] *= 2.
#         elif ch == '-':
#             result['gain'] *= 0.5
#         else:
#             print('\x1b[31;40m', usage_line.center(args.columns, '#'),
#                   '\x1b[0m', sep='')
#             return None, True
#     return result, True


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

    if kwargs['window'] > WINDOW_LENGTH:
        print("window must be <=", WINDOW_LENGTH)
        parser.exit(1)

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
        output_handler = MidiOutput(midi_port, midi_channel, verbose=verbose)

    if verbose:
        a = audio_input_info
        print(f"""
    Audio input:
        {sd.get_portaudio_version()[1]}
        {a['name']}:
            Sample rate: {a['default_samplerate']}
            Input latency: {(a['default_low_input_latency']*1000):.1f} - {(a['default_high_input_latency']*1000):.1f} ms
            Output latency: {(a['default_low_output_latency']*1000):.1f} - {(a['default_high_output_latency']*1000):.1f} ms

    Midi output:
        Port: {midi_port} channel: {midi_channel}"

    Model file: {model_file}
    """)

    with torch.no_grad():
        model = torch.load(model_file, map_location=ml_device).eval()

        with output_handler:
            audio_task = asyncio.create_task(
                transcribe_frame(
                    model=model,
                    window=kwargs['window'],
                    onset_threshold=kwargs['onset_threshold'],
                    frame_threshold=kwargs['frame_threshold'],
                    device=ml_device,
                    output=output_handler,
                    verbose=verbose))

            if verbose:
                print(f"Listening on {audio_input_info['name']}...")

            try:
                result, ok = await audio_task
                if ok == False:
                    sys.exit()
                # for key, value in result.items():
                #     if key == 'gain':
                #         gain *= value
            except asyncio.CancelledError:
                if verbose:
                    print('\nListening cancelled')
                return


def transcribe(model, audio, melspectrogram):

    mel = melspectrogram(audio.reshape(-1, audio.shape[-1])[:, :-1]).transpose(-1, -2)
    onset_pred, offset_pred, _, frame_pred, velocity_pred = model(mel)

    predictions = {
        'onset': onset_pred.reshape((onset_pred.shape[1], onset_pred.shape[2])),
        #'offset': offset_pred.reshape((offset_pred.shape[1], offset_pred.shape[2])),
        'frame': frame_pred.reshape((frame_pred.shape[1], frame_pred.shape[2])),
        'velocity': velocity_pred.reshape((velocity_pred.shape[1], velocity_pred.shape[2]))
    }
    #print("predict time:", 1000 * (end-start))
    return predictions


class MidiTransformer:
    def __init__(self, onset_threshold=0.5, frame_threshold=0.5):
        self.onsets = None
        self.frames = None
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold
        self.active_pitches = np.zeros(PITCHES, dtype='uint16')

    def extract_notes(self, predictions):
        """
        Finds the midi notes based on the onsets and frames information

        Parameters
        ----------
        predictions: dict containing:
            onset: torch.FloatTensor, shape = [frames, bins]
            offset: torch.FloatTensor, shape = [frames, bins]
            frame: torch.FloatTensor, shape = [frames, bins]
            velocity: torch.FloatTensor, shape = [frames, bins]

        Returns
        -------
        midi_messages: list of Midi.Messages
        """
        onsets = (predictions['onset'] > self.onset_threshold).cpu().to(torch.uint8)
        #offsets = (predictions['offset'] > self.onset_threshold).cpu().to(torch.uint8)
        frames = (predictions['frame'] > self.frame_threshold).cpu().to(torch.uint8)
        velocity = predictions['velocity']

        # add saved last frame values
        if self.onsets is not None:
            onsets = torch.cat([self.onsets, onsets[:, :]], dim=0)
        if self.frames is not None:
            frames = torch.cat([self.frames, frames[:, :]], dim=0)

        midi_messages = []

        # allow repeat onsets after this many frames
        # TODO: calculate this and add parameter
        min_onset_frame_gap = 10

        # TODO: optimize for mostly zero data
        # step through each new frame, look for onsets and ends of frames
        for pitch in range(PITCHES):
            velocity_samples = []
            note_on = False
            note_off = False
            note = MIN_MIDI + pitch

            for frame in range(1, onsets.shape[0]):
                onset = (onsets[frame,pitch].item() - onsets[frame-1,pitch].item()) == 1
                if onset:
                    if self.active_pitches[pitch] == 0 or self.active_pitches[pitch] > min_onset_frame_gap:
                        note_on = True
                        # NOTE: velocity frame doesn't have additional saved data so frame - 1 is correct
                        velocity_samples.append(velocity[frame - 1,pitch].item()) 
                        self.active_pitches[pitch] = 1

                elif self.active_pitches[pitch]:
                    off = frames[frame-1,pitch].item() - frames[frame,pitch].item() 
                    if off == 0:
                        self.active_pitches[pitch] += 1 # track number of frames this note has been on
                        velocity_samples.append(velocity[frame - 1,pitch].item())
                    elif off == 1:
                        note_off = True
                        self.active_pitches[pitch] = 0

                if note_on:
                    note_on = False
                    v = min(127, int(np.mean(velocity_samples) * 127))
                    midi_messages.append(
                        mido.Message('note_on', note=note, velocity=v))

                if note_off:
                    note_off = False
                    midi_messages.append(
                        mido.Message('note_off', note=note))


        # store last frames for next time
        self.onsets = onsets[-1:, :]
        self.frames = frames[-1:, :]

        return midi_messages


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=usage_line)

    parser.add_argument('-v', '--verbose', action='store_true', help='verbose')

    # model args
    parser.add_argument('model_file', nargs='?', type=str, default=None)
    parser.add_argument('-w', '--window', default=256, type=int)
    parser.add_argument('--onset-threshold', default=0.5, type=float)
    parser.add_argument('--frame-threshold', default=0.5, type=float)
    parser.add_argument('--ml-device', dest='ml_device', default='cuda' if torch.cuda.is_available() else 'cpu')

    # audio args
    parser.add_argument('-l', '--list-devices', action='store_true',
                        help='list audio devices and exit')
    parser.add_argument('-d', '--audio-device', type=int_or_str, dest='audio_device',
                        help='input device (numeric ID or substring)')
    # parser.add_argument('-g', '--gain', type=float, default=10,
    #                     help='initial gain factor (default %(default)s)')
    # parser.add_argument('-r', '--range', dest='freq_range', type=float, nargs=2,
    #                     metavar=('LOW', 'HIGH'), default=DEFAULT_FREQ_RANGE,
    #                     help='frequency range (default %(default)s Hz)')

    # midi output args
    parser.add_argument('-p', '--port', type=str, dest='midi_port',
                        help='midi port (string)')
    parser.add_argument('--channel', type=int, dest='midi_channel', default=1,
                        help='midi channel (default %(default)s)')

    # testing arguments
    parser.add_argument('-s', '--save-midi', type=str, default=None, dest='save_midi_file',
                        help='filename of midi file to save output to')

    args = parser.parse_args()

    try:
        asyncio.run(main(**vars(args)))
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')



