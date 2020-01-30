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

from aioconsole import ainput
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
FRAME_LENGTH = 256
# the minimum number of frames to treat as current data
MIN_FRAMES_TO_PROCESS = 2 

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
            print(f"{ms_since_start:8d}> {m}")

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
            self.saved_messages = []
            self.midi_file = os.path.expanduser(self.midi_file)
            if self.verbose:
                print(f"Saving to {self.midi_file}")
            if os.path.exists(self.midi_file):
                print(f"Warning: {self.midi_file} already exists. Overwriting...")

    async def send(self, messages):
        since_start = time.monotonic() - self.start
        for m in messages:
            m.channel = self.channel
            self.port.send(m)
            if self.verbose:
                print(f"{int(1000 * since_start):8d}: {self.port_name[:12]:<12}> {m}")
            if self.midi_file:
                self.saved_messages.append(m.copy(time = since_start))

    def close(self):
        self.port.close()

    def __enter__(self):
        self.port = mido.open_output(self.port_name)
        self.port.send(mido.Message('reset'))
        return self

    def __exit__(self, type, value, traceback):
        self.port.send(mido.Message('reset'))
        self.port.close()
        if self.midi_file:
            self.save_midi_file(self.midi_file)

    def save_midi_file(self, path):
        file = mido.MidiFile()
        track = mido.MidiTrack()
        file.tracks.append(track)
        ticks_per_second = file.ticks_per_beat * 2.0

        prev_time = 0
        for m in self.saved_messages:
            delta_ticks = int((m.time - prev_time) * ticks_per_second) # convert time to delta ticks
            track.append(m.copy(time=delta_ticks))
            prev_time = m.time

        if self.verbose:
            print(f"Saving midi to {path}")
        file.save(path)


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


async def transcribe_frame(model, output, 
    window_len=WINDOW_LENGTH, frame_len=FRAME_LENGTH,
    onset_threshold=0.5, frame_threshold=0.5, 
    device='cpu', verbose=False, gain=1., **kwargs):

    last_update = time.monotonic()

    # Buffer looks like this:
    # /------- window_len == buffer_len -----------\
    #                              /-- frame_len --\
    # [  previously predicted data |   new input   ]
    #
    # Each window of data we predict and roll the data one frame.
    # TODO: We then need to remove re-predicted messages.
    buffer_len = window_len
    buf = np.zeros(buffer_len, dtype='int16')
    temp_buf = np.zeros(buffer_len, dtype='int16')
    buf_frame_start = buffer_len - frame_len
    buf_end = buf_frame_start
    count = 0
    window_to_frame_ratio = window_len // frame_len
    ignore_frames = max(0, window_to_frame_ratio - MIN_FRAMES_TO_PROCESS)

    melspectrogram = MelSpectrogram(N_MELS, SAMPLE_RATE, window_len, frame_len, mel_fmin=MEL_FMIN, mel_fmax=MEL_FMAX, gain=gain)
    melspectrogram.to(device)

    transformer = MidiTransformer(onset_threshold, frame_threshold, verbose=verbose)

    # update frequency information when verbose = True
    report_freq = 100
    update_durations = [0 for _ in range(report_freq)]
    frame_lens = [0 for _ in range(report_freq)]
    report_count = 0

    async for indata, frame_count, status in inputstream_generator(**kwargs):
        if status:
            text = ' ' + str(status) + ' '
            print('\x1b[34;40m', text.center(columns, '#'),
                  '\x1b[0m', sep='')

        data = indata[:, 0]
        in_len = len(data)
        assert in_len == frame_count

        if in_len > frame_len:
            print(f"Buffer overrun: {in_len - frame_len}")
            in_len = frame_len
            data = data[:frame_len] # TODO: better to keep beginning or end?

        count += in_len

        if count < frame_len:
            # append into buffer
            buf[buf_end:(buf_end+in_len)] = data
            buf_end += in_len

        else:
            # fill up buffer
            remaining_buf = buffer_len - buf_end
            if remaining_buf > 0:
                buf[buf_end:buffer_len] = data[:remaining_buf]

            # note: buf is copied
            audio = torch.ShortTensor(buf).to(device)
            audio = audio.float().div_(32768.0)

            # predict and convert to midi
            # approx: 5ms to predict
            predictions = transcribe(model, audio, melspectrogram)
            # approx: 2ms to extract notes
            midi_messages = transformer.extract_notes(predictions, ignore_frames=ignore_frames)
            if len(midi_messages) > 0:
                # approx: < 0.1ms to send
                await output.send(midi_messages)

            # roll buffer 1 frame length
            if frame_len < buffer_len:
                #buf = np.roll(buf, -frame_len)
                # faster than roll (but more memory)
                temp_buf[:-frame_len] = buf[frame_len:]
                buf = temp_buf

            # reset buffer (with excess in data)
            count = in_len - remaining_buf
            buf_end = buf_frame_start + count
            buf[buf_frame_start:buf_end] = data[remaining_buf:]

            # track and report
            if verbose:
                frame_lens[report_count] = frame_count
                now = time.monotonic()
                update_durations[report_count] = now - last_update
                last_update = now
                report_count += 1
                if report_count >= report_freq:
                    report_count = 0
                    avg_update = sum(update_durations) / len(update_durations)
                    avg_lens = sum(frame_lens) / len(frame_lens)
                    print(f"""{int(1000 * avg_update)}ms between updates. Avg frame lengths: {avg_lens}""")


def transcribe(model, audio, melspectrogram):

    mel = melspectrogram(audio.reshape(-1, audio.shape[-1])[:, :-1]).transpose(-1, -2)
    onset_pred, _, _, frame_pred, velocity_pred = model(mel)

    predictions = {
        'onset': onset_pred.reshape((onset_pred.shape[1], onset_pred.shape[2])),
        'frame': frame_pred.reshape((frame_pred.shape[1], frame_pred.shape[2])),
        'velocity': velocity_pred.reshape((velocity_pred.shape[1], velocity_pred.shape[2]))
    }
    return predictions


class MidiTransformer:
    def __init__(self, onset_threshold=0.5, frame_threshold=0.5, verbose=False):
        self.onsets = None
        self.frames = None
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold
        self.active_pitches = np.zeros(PITCHES, dtype='uint16')
        self.ignore_frames = 0
        self.verbose = verbose

    def extract_notes(self, predictions, ignore_frames=0):
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
        frames = (predictions['frame'] > self.frame_threshold).cpu().to(torch.uint8)
        velocity = predictions['velocity'].cpu()

        # can only increase ignore frames
        if ignore_frames > self.ignore_frames:
            self.ignore_frames = ignore_frames

        # add saved last frame values
        if self.ignore_frames == 0 and ignore_frames == 0:
            if self.onsets is not None and self.frames is not None:
                ignore_frames = 1
                onsets = torch.cat([self.onsets, onsets[:, :]], dim=0)
                frames = torch.cat([self.frames, frames[:, :]], dim=0)
                # duplicate first frame of velocity to match onsets
                velocity = torch.cat([velocity[:1, :], velocity[:, :]], dim=0)

        assert ignore_frames < onsets.shape[0]

        midi_messages = []
        last_frame = onsets.shape[0] - 1

        # allow repeat onsets after this many frames
        # TODO: calculate this and add parameter
        min_onset_frame_gap = last_frame * 3

        # TODO: optimize for mostly zero data

        # step through each new frame, look for onsets and ends of frames
        for pitch in range(PITCHES):
            velocity_samples = []
            note_on = False
            note_off = False
            note = MIN_MIDI + pitch

            for frame in range(ignore_frames, onsets.shape[0]):
                this = onsets[frame,pitch].item()
                prev = onsets[frame-1,pitch].item()
                onset = (this - prev) == 1
                # check prev frame if looking for onset on the last frame
                # if not onset and self.onsets is not None and frame == last_frame:
                #     saved_onset = self.onsets[0,pitch].item()
                #     #saved_frame = self.frames[0,pitch].item()
                #     if prev != saved_onset:
                #         if self.verbose:
                #             print(f"onset mismatch({note}) c: {this}, p: {prev} != {saved}")
                #         if self.active_pitches[pitch] == 0:
                #             onset = ((this - saved) == 1) or ((prev - this) == 1)
                if onset:
                    if self.active_pitches[pitch] == 0 or self.active_pitches[pitch] > min_onset_frame_gap:
                        note_on = True
                        velocity_samples.append(velocity[frame,pitch].item()) 
                        self.active_pitches[pitch] = 1

                elif self.active_pitches[pitch] > 0:
                    # to be doubly sure, check last two frames (which will delay offsets by 1 frame)
                    off = frames[frame,pitch].item() == 0 and frames[frame-1,pitch].item() == 0
                    if off and self.active_pitches[pitch] > min_onset_frame_gap:
                        note_off = True
                        self.active_pitches[pitch] = 0
                    else:
                        velocity_samples.append(velocity[frame,pitch].item())
                        self.active_pitches[pitch] += 1 # track number of frames this note has been on

            if note_on:
                v = min(127, int(np.mean(velocity_samples) * 127))
                midi_messages.append(
                    mido.Message('note_on', note=note, velocity=v))

            if note_off:
                midi_messages.append(
                    mido.Message('note_off', note=note))

        # store last frames for next time
        # NOTE: this assumes that ignore_frames only increases
        self.onsets = onsets[-1:, :]
        self.frames = frames[-1:, :]

        return midi_messages


def parse_interactive_input(input, adjustable_params):
    usage_line = f"""Interactive commands:
    <setting/command> [=] [<value>]

    q, quit:\t\texit
    w, window = {adjustable_params['window_len']}:\tset the window size
    f, frame = {adjustable_params['frame_len']}:\tset the frame size 
    g, gain = {adjustable_params['gain']}:\tset the input gain
    on, onset = {adjustable_params['onset_threshold']}:\tset the onset threshold
    of, offset = {adjustable_params['frame_threshold']}:\tset the offset/frame threshold
    """

    input = input.lower()
    # remove optional =
    input = input.replace("=", " ")
    value = None
    if len(input) > 0: 
        # split into command and value
        result = input.split(None,1)
        if len(result) == 2:
            input, value = result[0], result[1]
    
    if input.startswith('q'):
        print("Quitting...")
        return None, False

    if value:
        if input.startswith('w'):
            print("Setting window to", value)
            adjustable_params['window_len'] = int(value)
            return adjustable_params, True

        if input.startswith('f'):
            print("Setting frame to", value)
            adjustable_params['frame_len'] = int(value)
            return adjustable_params, True

        if input.startswith('g'):
            print("Setting gain to", value)
            adjustable_params['gain'] = float(value)
            return adjustable_params, True

        if input.startswith('on'):
            print("Setting onset threshold to", value)
            adjustable_params['onset_threshold'] = float(value)
            return adjustable_params, True

        if input.startswith('of'):
            print("Setting offest/frame threshold to", value)
            adjustable_params['frame_threshold'] = float(value)
            return adjustable_params, True

    print(usage_line)
    return None, True


async def wait_for_input(adjustable_params):
    response = await ainput('> ')
    return parse_interactive_input(response, adjustable_params)


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
    gain=1.,
    model_file = None,
    ml_device = 'cpu',
    midi_port = None,
    midi_channel = 0,
    verbose = False,
    save_midi_file = None,
    interactive = False,
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

    # window and frame checks and warnings
    if kwargs['window'] < 1024:
        print(f"window ({kwargs['window']}) must be >= 1024")
        parser.exit(1)
    if kwargs['window'] > 2048:
        print(f"\n \033[1;31;40mWARNING:\033[0m window ({kwargs['window']}) should be <= 2048 for minimal latency")

    if kwargs['frame'] == 0: # default to quarter of window frame size
        kwargs['frame'] = kwargs['window'] // 4;
    if kwargs['frame'] < 256:
        print(f"\n \033[1;31;40mWARNING:\033[0m frame ({kwargs['frame']}) is likely too small")

    if kwargs['frame'] >= kwargs['window']:
        print(f"frame ({kwargs['frame']}) must be < window ({kwargs['window']})")
        parser.exit(1)
    if kwargs['window'] % kwargs['frame'] != 0:
        print(f"window ({kwargs['window']}) must be divisible by frame ({kwargs['frame']})")
        parser.exit(1)
    if kwargs['window'] // kwargs['frame'] != 4:
        print(f"\n \033[1;31;40mWARNING:\033[0m window ({kwargs['window']}) should be 4 times larger than frame ({kwargs['frame']})")


    audio_input_info = sd.query_devices(audio_device, 'input')
    if verbose:
        a = audio_input_info
        print(f"""
    Audio input: {sd.get_portaudio_version()[1]}
        Device: {a['name']}:
            Sample rate:    {a['default_samplerate']} downsampled to {SAMPLE_RATE}
            Input latency:  {(a['default_low_input_latency']*1000):.1f} - {(a['default_high_input_latency']*1000):.1f} ms
            Output latency: {(a['default_low_output_latency']*1000):.1f} - {(a['default_high_output_latency']*1000):.1f} ms
        Window size: {kwargs['window']}
        Frame size:  {kwargs['frame']}""")

    # construct output
    if midi_port is None:
        output_handler = Output()
        if verbose:
            print(f"""
    Output: console
        No midi output, display only. Use -p to set midi port.""")

    else:
        # find midi port match by name
        for name in mido.get_output_names():
            if midi_port in name:
                midi_port = name
                break

        output_handler = MidiOutput(midi_port, midi_channel, verbose=verbose, save_to=save_midi_file)
        if verbose:
            print(f"""
    Output: midi
        Port: {midi_port} channel: {midi_channel}""")

    if verbose:
        print(f"""
    Model: {model_file}
        Note on threshold:  {kwargs['onset_threshold']}
        Note off threshold: {kwargs['frame_threshold']}
    """)

    with torch.no_grad():
        model = torch.load(model_file, map_location=ml_device).eval()

        with output_handler:

            transcribe_params = dict(
                model=model,
                output=output_handler,
                window_len=kwargs['window'],
                frame_len=kwargs['frame'],
                onset_threshold=kwargs['onset_threshold'],
                frame_threshold=kwargs['frame_threshold'],
                device=ml_device,
                gain=gain,
                verbose=verbose
            )

            while True: # loop because of input task
                if interactive:
                    input_task = asyncio.create_task(wait_for_input(transcribe_params))
                audio_task = asyncio.create_task(transcribe_frame(**transcribe_params))

                if verbose:
                    print(f"Listening on {audio_input_info['name']}...")

                try:
                    if interactive:
                        result, ok = await wait_first(audio_task, input_task)
                    else:
                        result, ok = await audio_task

                    if ok == False:
                        sys.exit()
                    if result:
                        transcribe_params = result

                except asyncio.CancelledError:
                    if verbose:
                        print('\nListening cancelled')
                    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=usage_line)

    parser.add_argument('-v', '--verbose', action='store_true', help='verbose')

    # model args
    parser.add_argument('model_file', nargs='?', type=str, default=None)
    parser.add_argument('-w', '--window', default=WINDOW_LENGTH, type=int)
    parser.add_argument('-f', '--frame', default=0, type=int)
    parser.add_argument('--onset-threshold', default=0.5, type=float)
    parser.add_argument('--frame-threshold', default=0.5, type=float)
    parser.add_argument('--ml-device', dest='ml_device', default='cuda' if torch.cuda.is_available() else 'cpu')

    # audio args
    parser.add_argument('-l', '--list-devices', action='store_true',
                        help='list audio devices and exit')
    parser.add_argument('-d', '--audio-device', type=int_or_str, dest='audio_device',
                        help='input device (numeric ID or substring)')
    parser.add_argument('-g', '--gain', type=float, default=1.,
                        help='initial gain factor (default %(default)s)')
    # parser.add_argument('-r', '--range', dest='freq_range', type=float, nargs=2,
    #                     metavar=('LOW', 'HIGH'), default=DEFAULT_FREQ_RANGE,
    #                     help='frequency range (default %(default)s Hz)')

    # midi output args
    parser.add_argument('-p', '--port', type=str, dest='midi_port',
                        help='midi port (string)')
    parser.add_argument('-c', '--channel', type=int, dest='midi_channel', default=1,
                        help='midi channel (default %(default)s)')

    # testing arguments
    parser.add_argument('-s', '--save-midi', type=str, default=None, dest='save_midi_file',
                        help='filename of midi file to save output to')
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='display interactive console for changing settings')

    args = parser.parse_args()

    try:
        asyncio.run(main(**vars(args)))
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')



