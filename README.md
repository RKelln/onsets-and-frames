# PyTorch Implementation of Onsets and Frames

This is a [PyTorch](https://pytorch.org/) implementation of Google's [Onsets and Frames](https://magenta.tensorflow.org/onsets-frames) model, using the [Maestro dataset](https://magenta.tensorflow.org/datasets/maestro) for training and the Disklavier portion of the [MAPS database](http://www.tsi.telecom-paristech.fr/aao/en/2010/07/08/maps-database-a-piano-database-for-multipitch-estimation-and-automatic-transcription-of-music/) for testing.

## Instructions

This project is quite resource-intensive; 32 GB or larger system memory and 8 GB or larger GPU memory is recommended. 

### Downloading Dataset

The `data` subdirectory already contains the MAPS database. To download the Maestro dataset, first make sure that you have `ffmpeg` executable and run `prepare_maestro.sh` script:

```bash
ffmpeg -version
cd data
./prepare_maestro.sh
```

This will download the full Maestro dataset from Google's server and automatically unzip and encode them as FLAC files in order to save storage. However, you'll still need about 200 GB of space for intermediate storage.

### Training

All package requirements are contained in `requirements.txt`. To train the model, run:

```bash
pip install -r requirements.txt
python train.py
```

`train.py` is written using [sacred](https://sacred.readthedocs.io/), and accepts configuration options such as:

```bash
python train.py with logdir=runs/model iterations=1000000
```

Trained models will be saved in the specified `logdir`, otherwise at a timestamped directory under `runs/`.

### Testing

To evaluate the trained model using the MAPS database, run the following command to calculate the note and frame metrics:

```bash
python evaluate.py runs/model/model-100000.pt
```

Specifying `--save-path` will output the transcribed MIDI file along with the piano roll images:

```bash
python evaluate.py runs/model/model-100000.pt --save-path output/
```

In order to test on the Maestro dataset's test split instead of the MAPS database, run:

```bash
python evaluate.py runs/model/model-100000.pt Maestro test
```

### Real-time 

WIP: Very experimental and currently non-functional. Some basic install and use instructions:

NOTE: Uses python 3.7, thus for conda install something like this:

```bash
conda create -n py3_onsets python=3.7
conda activate py3_onsets
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
git clone https://github.com/RKelln/onsets-and-frames.git
cd onsets-and-frames
# currently realtime code lives in the realtime branch:
git checkout realtime
pip install -r requirements.txt
```

You will need to set up a synth and generally for testing you will need a virtual mic that you can send audio files to. Instructions for this are lengthy and complicated and depend on OS.

For Ubuntu 18.04 soemthing like this may work:

#### Midi playback

References:
    http://linux-audio.com/TiMidity-howto.html
    http://tedfelix.com/linux/linux-midi.html
    https://hatari.tuxfamily.org/doc/midi-linux.txt
    https://wiki.archlinux.org/index.php/FluidSynth

```bash
sudo apt install fluidsynth fluid-soundfont-gm pmidi alsa-utils

fluidsynth --audio-driver=alsa --midi-driver=alsa_seq /usr/share/sounds/sf2/FluidR3_GM.sf2
```

This sometimes produces distortion, this seems to help:
```bash
fluidsynth -a alsa -l -o audio.period-size=128  /usr/share/sounds/sf2/FluidR3_GM.sf2
```

See audio hardware:
```bash
aplay -l
```

See midi info:
```bash
aconnect -l
```

#### Fluidsynth 

Fluidsynth needs a few more thing to realy work correctly:
1. Create An "audio" Group

  First, let's check to see if your system already has an audio group:
```bash
grep audio /etc/group
> audio:x:29:pulse
```

If you see an "audio" line like the one above, then you've already got an audio group and you can skip to Group Limits.

If grep didn't find an audio group, add one with groupadd:
```bash
sudo groupadd audio
```

2. Group Limits

The limits for the audio group can usually be found in `/etc/security/limits.d/audio.conf`. Check to see if that file exists on your system. If not, create one. You might need to create the `limits.d` directory

Then create the audio.conf file in there. I usually use nano:
```bash
sudo nano /etc/security/limits.d/audio.conf
```

And add the following lines:
```
@audio   -  rtprio     95
@audio   -  memlock    unlimited
#@audio   -  nice       -19
```


Commands in fluidsynth:

List instruments:
```
> inst 0
```

Change instrument on channel:
```
> select 0 0 0 73
```
`select <chan> <sfont> <bank> <preset>`

    
#### Create a virtual mic

For this you probably need to create a wav file with a sample rate of 48000 (loopback wants that rate for some reason). Find a short bit of piano music, then convert to 48000 sample rate:

```bash
sox ~/Music/test_piano.wav -r 48000 ~/Music/piano_48000rate.wav
```

```bash
# create loopback
sudo modprobe snd-aloop

# check on audio devices:
aplay -l

# or:
python realtime_transcribe.py -l

# on my machine loopback was card 2
# for some reason the first loopback device doesn't work well, but the second does
$ ffmpeg -re -i ~/Music/piano_48000rate.wav -f s161e -f alsa hw:2,1
```

#### Putting it all together:
```bash    
# in tab 1 start fluidsynth:
fluidsynth -a alsa -l -o audio.period-size=128  /usr/share/sounds/sf2/FluidR3_GM.sf2
# note the port forfluidsynth
aconnect -l

# in tab 2: get the device number for the virtual mic (what is mapped to hw:2,1)
python realtime_transcribe.py -l

# -d for loopback device index, -p for fluidsynth port and channel:
python realtime_transcribe.py -d 7 -p 129:0 models/uni/model-1000000.pt

# in tab 3:
$ ffmpeg -re -i ~/Music/piano_48000rate.wav -f s161e -f alsa hw:2,1
```




## Implementation Details

This implementation contains a few of the additional improvements on the model that were reported in the Maestro paper, including:

* Offset head
* Increased model capacity, making it 26M parameters by default
* Gradient stopping of inter-stack connections
* L2 Gradient clipping of each parameter at 3
* Using the HTK mel frequencies

Meanwhile, this implementation does not include the following features:

* Variable-length input sequences that slices at silence or zero crossings
* Harmonically decaying weights on the frame loss

Despite these, this implementation is able to achieve a comparable performance to what is reported on the Maestro paper as the performance without data augmentation.


