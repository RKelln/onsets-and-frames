import sys

import numpy as np

from midi import parse_midi
from constants import *


# note requires Python 3.7 so that dict retains insertion order
def note_onsets_per_time(notes, quantization=0.15):
    """Returns a dict containing a quantized time and all notes played within that quantized time"""

    d = {}
    simultaneous_notes = []
    t = 0
    for n in notes:
        onset, offset, note, velocity = n
        # round onset to millisecond
        onset = round(onset, 4)
        if t == 0 or onset - t > quantization:
            t = onset 
            d.setdefault(t, [])
        d[t].append(int(note))

    return d


# times are inclusive
def notes_in_time(from_time, to_time, notes):
    """Returns all the notes in the notes dict (created by note_onsets_per_time) that are within the time range inclusively."""
    ret = []
    # NOTE: unoptimized
    for t, n in notes.items():
        if t >= from_time and t <= to_time:
            ret.extend(n)
    return ret


def compare_notes(true_notes, test_notes):
    """Compare two np.array of (onset, offset, note, velocity) rows. Ground truth versus test"""

    print(true_notes)

    ONSET = 0
    OFFSET = 1
    NOTE = 2
    VELOCITY = 3

    QUANTIZATION = 0.15
    LOOKBACK_SEC = 0.3
    LOOKAHEAD_SEC = 0.5

    perfect_matches = 0
    MIDI_NOTES = range(MIN_MIDI, MAX_MIDI+1)
    true_per_note = {note:[] for note in MIDI_NOTES}
    test_per_note = {note:[] for note in MIDI_NOTES}

    true_per_time = note_onsets_per_time(true_notes, 0.15)
    print(true_per_time)
    test_per_time = note_onsets_per_time(test_notes, LOOKBACK_SEC)
    print(test_per_time)


    # simple compare
    for i, n in enumerate(true_notes):
        onset, offset, note, velocity = n
        true_per_note[int(note)].append(n)

        if i < len(test_notes) and test_notes[i][NOTE] == note:
            perfect_matches += 1
    print(f"perfect matches: {perfect_matches}")

    for i, n in enumerate(test_notes):
        test_per_note[int(n[NOTE])].append(n)

    # per note matching
    #print("true counts:")
    for midi_note, all_notes in true_per_note.items():
        print(f"{midi_note}: {len(all_notes)}", end=" ")
    #print()

    #print("test counts:")
    for midi_note, all_notes in test_per_note.items():
        print(f"{midi_note}: {len(all_notes)}", end=", ")
    #print()

    # grouping per time
    # note Python 3.7+ required for ordered dicts
    time_errors = {}
    missing_note_errors = {}
    correct_notes = {}
    first_time_true = next(iter( true_per_time.keys() ))
    first_time_test = next(iter( test_per_time.keys() ))
    offset = first_time_test - first_time_true
    print(f"First note offset: {offset}s")
    for t, n in true_per_time.items():
        t = round(t + offset, 4)
        actual = notes_in_time(t - LOOKBACK_SEC, t + LOOKAHEAD_SEC, test_per_time)
        # simple set comparison (find any missing notes)
        missing = set(n) - set(actual)
        if len(missing) > 0:
            for mn in missing:
                missing_note_errors.setdefault(mn, []).append({"time": t, "found": actual})
            time_errors[t] = {"missing": missing, "found": actual}
            print(f"missing notes at {t:.4f}: {time_errors[t]}")
        
        for cn in set(n) & set(actual):
            correct_notes.setdefault(cn, []).append(t)

    print(f"Correct notes: {correct_notes}")

    # simple report
    total_difference = 0
    total_notes = 0
    for n in MIDI_NOTES:
        # by simple order
        total_notes += len(true_per_note[n])
        count_test = len(test_per_note.get(n, []))
        count_true = len(true_per_note.get(n, []))
        count_diff = count_test - count_true 
        total_difference += abs(count_diff)

        # grouped by time
        time_correct = len(correct_notes.get(n, []))
        time_missing = len(missing_note_errors.get(n, []))
        time_diff = time_correct - time_missing 

        print(f"{n:3d}: {count_true:2d}:{count_test:2d} {count_diff:+3d}",end=',')
        print(f"  by time: {time_correct:2d}:{time_missing:2d} {time_diff:+3d}")
    
    percent = ((total_notes - total_difference) / total_notes) * 100.
    print(f"total difference: {total_difference} {percent:.2f}% correct")


if __name__ == '__main__':

    if len(sys.argv) < 3:
        "Usage: python calibration.py true_notes.mid heard_notes.mid"

    true_notes = parse_midi(sys.argv[1])
    test_notes = parse_midi(sys.argv[2])
    
    compare_notes(true_notes, test_notes)

