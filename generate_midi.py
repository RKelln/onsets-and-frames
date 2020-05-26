#!/usr/bin/env python3
"""
Generate a suite of semi-random midi sequences that test a full range of piano expression.

You need Python 3.7 or newer to run this.

Exmaple:

    $ python generate_midi.py -o <output dir> -s <random seed int> -i <generation script> 
"""

import argparse
import numbers
import os
import random

from pathlib import Path, PurePath
from collections.abc import Sequence

from mido import Message, MidiFile, MidiTrack

SEED = 123456789

MIN_NOTE = 21
MAX_NOTE = 108
MIN_VEL = 10
MAX_VEL = 127
# time between notes in seconds
MIN_TEMPO = 0.1
MAX_TEMPO = 3.
TEMPO_STEP = 0.1
# how long notes are held in seconds
MIN_SUSTAIN = 0.1
MAX_SUSTAIN = 10.
SUSTAIN_STEP = 0.2
# number of notes active at the same time
MIN_POLYPHONY = 1
MAX_POLYPHONY = 20
# silence in seconds before first note
START_SILENCE = 1.0

# you can create your own generator scripts to pass in
example_generation_script = """
messages = generate_sequence(500, 
  note_range=range(MIN_NOTE, MAX_NOTE),
  velocity_range=range(MIN_VEL, MAX_VEL),
  tempo_range=linrange(0, 0.5, 0.1),
  sustain_range=linrange(MIN_SUSTAIN, MAX_SUSTAIN, SUSTAIN_STEP),
  polyphony_range=range(5, 20),
  start_time=START_SILENCE,
  verbose=verbose
)
save_midi(messages, output_path / "generated_example_1.mid")
"""

# from http://code.activestate.com/recipes/579000/
class linspace(Sequence):
    """linspace(start, stop, num) -> linspace object
    
    Return a virtual sequence of num numbers from start to stop (inclusive).
    
    If you need a half-open range, use linspace(start, stop, num+1)[:-1].
    """
    
    def __init__(self, start, stop, num):
        if not isinstance(num, numbers.Integral) or num <= 1:
            raise ValueError('num must be an integer > 1')
        self.start, self.stop, self.num = start, stop, num
        self.step = (stop-start)/(num-1)
    def __len__(self):
        return self.num
    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[x] for x in range(*i.indices(len(self)))]
        if i < 0:
            i = self.num + i
        if i >= self.num:
            raise IndexError('linspace object index out of range')
        if i == self.num-1:
            return self.stop
        return self.start + i*self.step
    def __repr__(self):
        return '{}({}, {}, {})'.format(type(self).__name__,
                                       self.start, self.stop, self.num)
    def __eq__(self, other):
        if not isinstance(other, linspace):
            return False
        return ((self.start, self.stop, self.num) ==
                (other.start, other.stop, other.num))
    def __ne__(self, other):
        return not self==other
    def __hash__(self):
        return hash((type(self), self.start, self.stop, self.num))    

class linrange(Sequence):
    """linrange(start, stop, step) -> linrange object
    
    Return a virtual sequence of numbers from start to stop (inclusive).
    """
    
    def __init__(self, start, stop, step):
        if not isinstance(step, numbers.Real) or step <= 0:
            raise ValueError('step must be a real number > 0')
        self.start, self.stop, self.step = start, stop, step
        self.num = int((stop-start)/step)
    def __len__(self):
        return self.num
    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[x] for x in range(*i.indices(len(self)))]
        if i < 0:
            i = self.num + i
        if i >= self.num:
            raise IndexError('linspace object index out of range')
        if i == self.num-1:
            return self.stop
        return self.start + i*self.step
    def __repr__(self):
        return '{}({}, {}, {}) len={}'.format(type(self).__name__,
                                       self.start, self.stop, self.step, self.num)
    def __eq__(self, other):
        if not isinstance(other, linspace):
            return False
        return ((self.start, self.stop, self.step) ==
                (other.start, other.stop, other.step))
    def __ne__(self, other):
        return not self==other
    def __hash__(self):
        return hash((type(self), self.start, self.stop, self.step))  

SCALE_STEPS = [2,2,1,2,2,2,1] # note steps
NOTE_NAMES = ["C", "B", "Bb", "A", "Ab", "G", "Gb", "F", "E", "Eb","D", "Db"]
STARTING_NOTES = [24, 23, 22, 21, 32, 31, 30, 29, 28, 27, 26, 25]
SCALE_NOTES = []
for starting_note in STARTING_NOTES:
  notes = []
  note = starting_note
  notes.append(note)
  scale_steps_index = 0
  steps_len = len(SCALE_STEPS)
  while note < MAX_NOTE:
    note += SCALE_STEPS[scale_steps_index % steps_len]
    scale_steps_index += 1
    if note <= MAX_NOTE:
      notes.append(note)
  SCALE_NOTES.append(notes)

def generate_sequence(
  length=500, 
  note_range=range(MIN_NOTE, MAX_NOTE),
  velocity_range=range(MIN_VEL, MAX_VEL),
  tempo_range=linrange(MIN_TEMPO, MAX_TEMPO, TEMPO_STEP),
  sustain_range=linrange(MIN_SUSTAIN, MAX_SUSTAIN, SUSTAIN_STEP),
  polyphony_range=range(MIN_POLYPHONY, MAX_POLYPHONY),
  start_time=0.,
  verbose = False,
  note_step=None):
  
  midi = []
  current_notes = []
  note_index = None
  current_end_times = []
  poly = random.choice(polyphony_range)
  current_time = start_time
  last_poly_change = current_time
  if verbose:
    print(".", end = '', flush=True)

  while int(current_time) < length and len(midi) < length:
    if len(midi) % 100 == 0 and verbose:
      print(".", end = '', flush=True)

    if note_step is None or note_index is None:
      note_index = random.randrange(len(note_range))
    else:
      step = note_step
      if not isinstance(note_step, int):
        step = random.choice(note_step)
      # step through the note_range
      note_index += step
      # loop on overrun
      if note_index < 0:
        note_index = len(note_range) + note_index
      elif note_index >= len(note_range):
        note_index = note_index - len(note_range) 

    note = note_range[note_index]
    # ensure that we aren't out of bounds
    if note < MIN_NOTE or note > MAX_NOTE:
      continue

    if len(current_notes) == 0 or (note not in current_notes and len(current_notes) < poly):
      # add a new note
      velocity = random.choice(velocity_range)
      duration = random.choice(sustain_range)
      if len(current_notes) == 0 or len(current_notes) >= poly - 1 or current_time - last_poly_change > 10:
        poly = random.choice(polyphony_range)
        last_poly_change = current_time

      end = current_time + duration
      current_notes.append(note)
      current_end_times.append(end)
      midi.append(Message("note_on", note=note, velocity=velocity, time=current_time))
      midi.append(Message("note_off", note=note, velocity=velocity, time=end))

    # advance time (if no more notes at the same time)
    count = len(current_notes)
    if count >= poly or random.randint(count, poly+1) >= poly:
      current_time += random.choice(tempo_range)
      #print(current_time, len(current_notes), len(current_end_times), poly)

    # go through ending times and remove notes that have ended (backwards to be safe)
    for i in reversed(range(len(current_end_times))):
      if current_time > current_end_times[i]:
        del current_end_times[i]
        del current_notes[i]

  if verbose:
    print(f"length ({length}): {len(midi)} notes, duration: {current_time:1.0f}s")
  
  return midi

def save_midi(messages, filepath):

  # convert absolute times into relative times for midi file format
  messages.sort(key=lambda message: message.time)

  file = MidiFile()
  track = MidiTrack()
  file.tracks.append(track)
  ticks_per_second = file.ticks_per_beat * 2.0
  last_tick = 0
  for m in messages:
      current_tick = int(m.time * ticks_per_second)
      # ensure a minimum start silence
      if current_tick <= 0:
        current_tick = int(START_SILENCE * ticks_per_second)
      m.time = current_tick - last_tick
      track.append(m)
      last_tick = current_tick

  file.save(filepath)


def main(seed=SEED, verbose=False, output_path='.', generator_script=None):

  random.seed(seed)

  output_path = Path(output_path).expanduser()
  output_path.mkdir(parents=True, exist_ok=True)

  if generator_script is None:
    file_name = "generated.mid"
    messages = generate_sequence(
      length=500, 
      note_range=range(MIN_NOTE, MAX_NOTE),
      velocity_range=range(MIN_VEL, MAX_VEL),
      tempo_range=linrange(MIN_TEMPO, MAX_TEMPO, TEMPO_STEP),
      sustain_range=linrange(MIN_SUSTAIN, MAX_SUSTAIN, SUSTAIN_STEP),
      polyphony_range=range(MIN_POLYPHONY, MAX_POLYPHONY),
      start_time=START_SILENCE,
      verbose=verbose
    )
    save_midi(messages, output_path / file_name)
    return

  # run the generator script
  if generator_script.endswith(".py"):
    # load from file
    generator_script = Path(generator_script).expanduser().read_text()
  
  exec(generator_script, globals(), locals())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate a suite of semi-random midi sequences that test a full range of piano expression.")

    parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
    parser.add_argument('-s', '--seed', default=SEED, type=int, help='Random seed')

    parser.add_argument('-o', '--output', type=str, default='.', dest='output_path',
                        help='file path for midi output')

    parser.add_argument('-i', '--input', type=str, default=example_generation_script, dest='generator_script',
                        help='script to use to geenrate midi')

    args = parser.parse_args()

    main(**vars(args))



