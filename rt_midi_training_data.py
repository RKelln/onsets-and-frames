versions = 10
length = 1000
params = dict(
    length=length, 
    note_range=range(MIN_NOTE, MAX_NOTE),
    velocity_range=range(10, 127),
    tempo_range=linrange(0.1, 0.3, 0.05),
    sustain_range=linrange(0.1, 3.0, 0.1),
    polyphony_range=range(5, 30),
    start_time=0.,
    verbose=verbose
)

def fast(params) -> list:
  p = dict(params)
  p['tempo_range'] = linrange(0.05, 0.4, 0.05)
  p['polyphony_range'] = range(1, 15)
  p['sustain_range'] = [0.1, 0.1, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
  p['length'] = p['length'] * 2
  return generate_sequence(**p)

def slow(params) -> list: 
  p = dict(params)
  p['tempo_range'] = linrange(0.1, 2.1, 0.5)
  p['polyphony_range'] = range(15, 40)
  p['sustain_range'] = linrange(1.0, 10., 0.25)
  p['length'] = p['length'] // 2
  return generate_sequence(**p)

for i in range(1, versions+1):
  # full range of key board and options
  p = dict(params)
  m = generate_sequence(**p)
  save_midi(m, output_path / f"full_{i:03d}.mid")
  save_midi(fast(p), output_path / f"fast_full_{i:03d}.mid")
  save_midi(slow(p), output_path / f"slow_full_{i:03d}.mid")

  # each octave:
  for note_min in range(0, 88, 8):
    note_min += MIN_NOTE
    note_max = note_min + 8
    p = dict(params)
    p['note_range'] = range(note_min, note_max)
    m = generate_sequence(**p)
    save_midi(m, output_path / f"range_{note_min}-{note_max}_{i:03d}.mid")

    save_midi(fast(p), output_path / f"fast_range_{note_min}-{note_max}_{i:03d}.mid")
    p['velocity_range'] = range(10,64)
    save_midi(fast(p), output_path / f"quiet_fast_range_{note_min}-{note_max}_{i:03d}.mid")

    p['velocity_range'] = params['velocity_range']
    save_midi(slow(p), output_path / f"slow_range_{note_min}-{note_max}_{i:03d}.mid")
    p['velocity_range'] = range(10,64)
    save_midi(slow(p), output_path / f"quiet_slow_range_{note_min}-{note_max}_{i:03d}.mid")

  # super poly
  m = []
  for note_min in range(0, 88, 16):
    note_min += MIN_NOTE
    note_max = note_min + 16
    p = dict(params)
    p['note_range'] = range(note_min, note_max)
    p['polyphony_range'] = range(2, 10)
    m.extend(slow(p))
  save_midi(m, output_path / f"poly_{i:03d}.mid")

  # slow and fast
  m = []
  p = dict(params)
  p['note_range'] = range(MIN_NOTE, MIN_NOTE+36)
  m.extend(slow(p))
  p['note_range'] = range(MIN_NOTE+36, MAX_NOTE)
  m.extend(generate_sequence(**p))
  save_midi(m, output_path / f"slow_med_{i:03d}.mid")

  m = []
  p = dict(params)
  p['note_range'] = range(MIN_NOTE, MIN_NOTE+24)
  m.extend(slow(p))
  p['note_range'] = range(MIN_NOTE+24, MAX_NOTE-24)
  m.extend(generate_sequence(**p))
  p['note_range'] = range(MAX_NOTE-24, MAX_NOTE)
  m.extend(fast(p))
  save_midi(m, output_path / f"slow_med_fast_{i:03d}.mid")

  # chords
  notes = ["C", "B", "Bf", "A", "Af", "G", "Gf", "F", "E", "Ef","D", "Df"]
  p = dict(params)
  p['note_range'] = [36, 40, 43, 48, 52, 55, 60, 64, 67, 72, 76, 79, 84, 88, 91, 96, 100, 103, 108]
  for note in notes:
    save_midi(generate_sequence(**p), output_path / f"full_{note}_{i:03d}.mid")
    save_midi(fast(p), output_path / f"fast_full_{note}_{i:03d}.mid")
    save_midi(slow(p), output_path / f"slow_full_{note}_{i:03d}.mid")
    save_midi(slow(p) + generate_sequence(**p), output_path / f"slow_fast_full_{note}_{i:03d}.mid")
    p['note_range'] = [p - 1 for p in p['note_range']]