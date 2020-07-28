#!/usr/bin/env python3
"""
Realtime audio piano audio transcription.

You need Python 3.7 or newer to run this.

List audio devices:

    $ python realtime_transcribe.py -l

Example:

    $ python -O realtime_transcribe.py -d 7 -p 129:0 models/uni/model-1000000.pt
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
from pythonosc.udp_client import SimpleUDPClient
import numpy as np
import sounddevice as sd
import soundfile
import mido
import rtmidi

from onsets_and_frames import *

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

    async def asend(self, messages, silent=False):
        self.send(messages, silent)

    def send(self, messages, silent=False):
        if silent: return
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
    def __init__(self, midi_port, midi_channel=1, save_to=None, verbose=False):
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

    def send(self, messages, silent=False):
        since_start = time.monotonic() - self.start
        for m in messages:
            if hasattr(m, 'channel'):
                m.channel = self.channel
            self.port.send(m)
            if not silent and self.verbose:
                print(f"{int(1000 * since_start):8d}: {self.port_name[:12]:<12}> {m}")
            if self.midi_file:
                self.saved_messages.append(m.copy(time = since_start))

    def close(self):
        self.reset()
        self.port.close()
        self.port = None

    def reset(self):
        if self.port:
            messages = [mido.Message('note_off', note = note) for note in range(PITCHES)]
            messages.append(mido.Message('reset'))
            self.send(messages, silent=True)

    def __enter__(self):
        self.port = mido.open_output(self.port_name)
        self.reset()
        return self

    def __exit__(self, type, value, traceback):
        self.close()
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


class OSCOutput(Output):
    """Sends midi messages over OSC"""

    def __init__(self, ip, port, midi_port=128, midi_channel=1, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.ip = ip
        self.port = int(port)
        self.midi_channel = min(15, max(0, midi_channel - 1))
        self.midi_port = midi_port

    def send(self, messages, silent=False):
        since_start = time.monotonic() - self.start
        for m in messages:
            if hasattr(m, 'channel'):
                m.channel = self.midi_channel
            # Midi messages are designated in python-osc as tuples of length 4
            #   port_id, status_byte, note, velocity
            # Mido.bytes returns a list of ints: 
            #   status_byte, note, velocity
            bytes = m.bytes()
            bytes.insert(0,self.midi_port)
            self.client.send_message("/midi", tuple(bytes))
            if not silent and self.verbose:
                print(f"{int(1000 * since_start):8d}: {self.ip}:{self.port}> {m}")

    def close(self):
        self.reset()
        self.client = None

    def reset(self):
        if self.client:
            messages = [mido.Message('note_off', note = note) for note in range(PITCHES)]
            messages.append(mido.Message('reset'))
            self.send(messages, silent=True)

    def __enter__(self):
        self.client = SimpleUDPClient(self.ip, self.port)
        self.reset()
        return self

    def __exit__(self, type, value, traceback):
        self.close()


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
    device='cpu', verbose=False, gain=1., debug=False, **kwargs):

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
    # mel spectrum always works best with this ration
    MAGIC_MEL_RATIO = 4
    hop_length = window_len // MAGIC_MEL_RATIO
    ignore_frames = max(0, MAGIC_MEL_RATIO - MIN_FRAMES_TO_PROCESS)

    melspectrogram = MelSpectrogram(N_MELS, SAMPLE_RATE, window_len, hop_length, mel_fmin=MEL_FMIN, mel_fmax=MEL_FMAX, gain=gain)
    melspectrogram.to(device)

    transformer = MidiTransformer(onset_threshold, frame_threshold, verbose=verbose, debug=debug)

    # update frequency information when debug = True
    report_freq = 5. # seconds
    last_report = time.monotonic() - report_freq + 1. # first report after 1 sec
    report_len = 400 # how many stats to store (roughly 5 sec at 13ms updates)
    update_durations = [0 for _ in range(report_len)]
    frame_lens = [0 for _ in range(report_len)]
    perf_notes = [0 for _ in range(report_len)]
    perf_predict = [0 for _ in range(report_len)]
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
            print(f"Buffer overrun: {in_len - frame_len}. Try --frame={in_len}.")
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
            # approx: 6ms to predict
            if debug: before_pred = time.perf_counter()
            predictions = transcribe(model, audio, melspectrogram)
            # approx: 1.5ms to extract notes
            if debug: before_notes = time.perf_counter()
            midi_messages = transformer.extract_notes(predictions, ignore_frames=ignore_frames)
            if debug: after_notes = time.perf_counter()

            if len(midi_messages) > 0:
                # approx: < 0.1ms to send
                await output.asend(midi_messages)

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
            if debug:
                now = time.monotonic()
                # loop the stats
                if report_count >= report_len: report_count = 0

                update_durations[report_count] = now - last_update
                frame_lens[report_count] = frame_count
                perf_predict[report_count] = before_notes - before_pred
                perf_notes[report_count] = after_notes - before_notes

                last_update = now
                report_count += 1

                if now - last_report > report_freq:
                    last_report = now
                    max_update = max(update_durations) * 1000
                    avg_update = sum(update_durations) / len(update_durations) * 1000
                    max_lens = max(frame_lens)
                    avg_lens = int(sum(frame_lens) / len(frame_lens))
                    max_predict = max(perf_predict) * 1000
                    avg_predict = sum(perf_predict) / len(perf_predict) * 1000
                    max_notes = max(perf_notes) * 1000
                    avg_notes = sum(perf_notes) / len(perf_notes) * 1000
                    print(f"""Perf:        <avg> - <max>
    predict: {avg_predict:>5.2f} - {max_predict:>5.2f} ms
    notes:   {avg_notes:>5.2f} - {max_notes:>5.2f} ms
    update:  {avg_update:>5.2f} - {max_update:>5.2f} ms
    input size: {avg_lens} - {max_lens}""")


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
    def __init__(self, onset_threshold=0.5, frame_threshold=0.5, verbose=False, debug=False):
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
        midi_messages = []

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

        last_frame = onsets.shape[0] - 1

        # allow repeat onsets after this many frames
        # TODO: calculate this and add parameter
        min_onset_frame_gap = last_frame * 3
        velocity_samples = []

        # step through each new frame, look for onsets and ends of frames
        for pitch in range(PITCHES):
            # optimize for mostly zero data
            if self.active_pitches[pitch] == 0 and not torch.any(onsets[:,pitch]):
                continue

            velocity_samples.clear()
            note_on = False
            note_off = False
            note = MIN_MIDI + pitch
            prev_onset = onsets[ignore_frames-1,pitch].item()
            prev_off = frames[ignore_frames-1,pitch].item()

            for frame in range(ignore_frames, onsets.shape[0]):
                this = onsets[frame,pitch].item()
                onset = (this - prev_onset) == 1
                prev_onset = this

                if onset and (self.active_pitches[pitch] == 0 or self.active_pitches[pitch] > min_onset_frame_gap):
                    note_on = True
                    velocity_samples.append(velocity[frame,pitch].item())
                    self.active_pitches[pitch] = 1

                elif self.active_pitches[pitch] > 0:
                    # to be doubly sure, check last two frames (which will delay offsets by 1 frame)
                    this = frames[frame,pitch].item()
                    off = this == 0 and prev_off == 0
                    prev_off = this

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

        # store last frames for next time if we aren't ignoring any frames
        # NOTE: this assumes that ignore_frames only increases
        if self.ignore_frames == 0:
            self.onsets = onsets[-1:, :]
            self.frames = frames[-1:, :]

        return midi_messages


async def parse_interactive_input(input, params):
    usage_line = f"""Interactive commands:
    <setting/command> [=] [<value>]

    q, quit:\t\texit
    r, reset:\t\treset midi
    v, verbose:\t\ttoggle verbose (currently: {params['verbose']})
    d, debug:\t\ttoggle debug mode (currently: {params['debug']})

    w, window = {params['window_len']}:\tset the window size
    f, frame = {params['frame_len']}:\tset the frame size 
    g, gain = {params['gain']}:\tset the input gain
    on, onset = {params['onset_threshold']}:\tset the onset threshold
    of, offset = {params['frame_threshold']}:\tset the offset/frame threshold
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

    if input.startswith('r'):
        print("Resetting midi...")
        params['output'].reset()
        return None, True

    if input.startswith('v'):
        print("Toggling verbose...")
        params['verbose'] = not params['verbose']
        return params, True

    if input.startswith('d'):
        print("Toggling debug information...")
        params['debug'] = not params['debug']
        return params, True

    try:
        if value:
            if input.startswith('w'):
                params['window_len'] = int(value)
                ok = check_window_frame(params['window_len'], params['frame_len'])
                if not ok:
                    return None, True
                print("Setting window to", value)
                return params, True

            if input.startswith('f'):
                params['frame_len'] = int(value)
                ok = check_window_frame(params['window_len'], params['frame_len'])
                if not ok:
                    return None, True
                print("Setting frame to", value)
                return params, True

            if input.startswith('g'):
                params['gain'] = float(value)
                print("Setting gain to", value)
                return params, True

            if input.startswith('on'):
                params['onset_threshold'] = float(value)
                print("Setting onset threshold to", value)
                return params, True

            if input.startswith('of'):
                params['frame_threshold'] = float(value)
                print("Setting offest/frame threshold to", value)
                return params, True
    except:
        print("Invalid input")

    print(usage_line)
    return None, True


async def wait_for_input(adjustable_params):
    response = await ainput('> ')
    return await parse_interactive_input(response, adjustable_params)


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


def warning(msg):
    return f"\033[1;31;93mWARNING:\033[0m {msg}"


def error(msg):
    return f"\033[1;31;40mERROR:\033[0m {msg}"


def check_window_frame(window, frame):
    """Returns true if dimensions are OK, false otherwise"""
    if window < 1024:
        print(error(f"window ({window}) must be >= 1024"))
        return False

    if window > 2048:
        print(warning(f"window ({window}) should be <= 2048 for minimal latency"))
    if frame < 256:
        print(warning(f"frame ({frame}) may cause overruns"))

    if frame >= window:
        print(error(f"frame ({frame}) must be < window ({window})"))
        return False

    if window % frame != 0:
        print(warning(f"window ({window}) not divisible by frame ({frame})"))
    if window // frame != 4:
        print(warning(f"window ({window}) not 4 times frame size ({frame})"))

    return True


async def main(list_devices=None, audio_device=None,
    gain=1.,
    model_file = None,
    ml_device = 'cpu',
    midi_port = None,
    midi_channel = 0,
    verbose = False,
    save_midi_file = None,
    interactive = False,
    debug=False,
    osc_address = None,
    **kwargs):

    if list_devices:
        print("Audio input available:")
        print(sd.query_devices())
        print("\nMidi output available:")
        print(mido.get_output_names())
        parser.exit(0)

    if model_file is None:
        print(error("Must supply the name of the model file"))
        parser.exit(1)

    if not os.path.exists(model_file):
        print(error("Cannot find model file:", model_file))
        parser.exit(1)

    # window and frame checks and warnings
    if kwargs['frame'] == 0: # default to quarter of window frame size
        kwargs['frame'] = kwargs['window'] // 4;
    check_window_frame(kwargs['window'], kwargs['frame'])

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
    if midi_port is None and osc_address is None:
        output_handler = Output()
        if verbose:
            print(f"""
    Output: console
        No midi output, display only. Use -p to set midi port.""")

    else:
        if osc_address:
            try:
                if midi_port is None: raise ValueError 
                midi_port = int(midi_port) # midi_port is a number for OSC
            except ValueError:
                print(error("Must set midi port using -p <number>"))
                parser.exit(1)

            ip, port = osc_address.split(":")
            ip = ip.strip()
            port = port.strip()
            output_handler = OSCOutput(ip, port, midi_port, midi_channel, verbose=verbose)
            if verbose:
                print(f"""
    Output: OSC
        Address: {ip} port: {port} channel: {midi_channel}""")

        else: # use midi
            # midi port here is the full name
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
                verbose=verbose,
                debug=debug
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
                        output_handler.verbose = transcribe_params['verbose']

                except asyncio.CancelledError:
                    if verbose:
                        print('\nListening cancelled')
                    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

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

    # OSC put args
    parser.add_argument('-o', '--osc', type=str, dest='osc_address', default=None,
                        help='Open Sound Control address (e.g.: 127.0.0.1:4000)')

    # testing arguments
    parser.add_argument('-s', '--save-midi', type=str, default=None, dest='save_midi_file',
                        help='filename of midi file to save output to')
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='display interactive console for changing settings')
    parser.add_argument('--debug', action='store_true',
                        help='display real-time debugging information')

    args = parser.parse_args()

    try:
        asyncio.run(main(**vars(args)))
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')


