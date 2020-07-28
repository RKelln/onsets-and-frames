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

#### Models

Download the model here:
[Realtime unidirectional model](https://drive.google.com/open?id=18VEiJAb4CKRSo_FZAcPC6g9A_sD6X8rY)

You can also try this model that isn't as optimized for realtime use:
[Unidirectional model](https://drive.google.com/open?id=19vDyiVoQDZ-B0KGOdOS78vIsWxgJbw-0)

#### Installation

NOTE: Uses python 3.7, thus when using `conda`, something like this should work:

```bash
conda create -n py3_onsets python=3.7
conda activate py3_onsets
conda install pytorch=1.3.1 torchvision=0.4.2 cudatoolkit=10.1 -c pytorch
git clone https://github.com/RKelln/onsets-and-frames.git
cd onsets-and-frames
# currently realtime code lives in the realtime branch:
git checkout realtime
pip install -r requirements.txt
```

You will need to set up a synth and generally for testing you will need a virtual mic that you can send audio files to. Instructions for this are lengthy and complicated and depend on OS.

For Ubuntu 18.04 something like the following may work.

#### Midi playback

References:

* http://linux-audio.com/TiMidity-howto.html
* http://tedfelix.com/linux/linux-midi.html
* https://hatari.tuxfamily.org/doc/midi-linux.txt
* https://wiki.archlinux.org/index.php/FluidSynth

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

Fluidsynth needs a few more thing to really work correctly:

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
ffmpeg -re -i ~/Music/piano_48000rate.wav -f s161e -f alsa hw:2,1
```


### Noisey dataset

Adding dynamic noise to the dataset requires subproceses that can take a lot of memory, if you run out of memory you can allocate an additional swap:

```bash
sudo fallocate -l 8G /path/to/swapfile
sudo chmod 600 /path/to/swapfile 
sudo mkswap /path/to/swapfile
sudo swapon /path/to/swapfile
```

#### Putting it all together:
```bash    
# in tab 1 start fluidsynth:
fluidsynth -a alsa -l -o audio.period-size=128  /usr/share/sounds/sf2/FluidR3_GM.sf2
# note the port for fluidsynth
aconnect -l

# in tab 2: get the device number for the virtual mic (what is mapped to hw:2,1)
python realtime_transcribe.py -l

# -d for loopback device index, -p for fluidsynth port and channel:
python -O realtime_transcribe.py -d 7 -p 129:0 onsets_uni_model_rt-1.pt

# in tab 3:
ffmpeg -re -i ~/Music/piano_48000rate.wav -f s161e -f alsa hw:2,1
```

In general something like this should work for sending OSC midi (typically what you'd use to connect to another program like Max/MSP):
    -O optimization flag turns off the asserts
    -i flag turns on interactive mode
    -o OSC output address (sends to 127.0.0.1:1234/midi)
    -p midi port (additional info sent through OSC)
    -c midi channel
    uses default audio input
```bash
python -O realtime_transcribe.py -i -o 127.0.0.1:1234 -p 128 -c 1  onsets_uni_model_rt-1.pt
```

For my system the optimal window and frame settings ended up being 2048 and 139 (where 139 was the largest frame sie observed without overruns):
```bash
python -O realtime_transcribe.py -i -o 127.0.0.1:1234 -p 128 -c 1 -w 2048 -f 139  onsets_uni_model_rt-1.pt
```
You can see the performance effects of different window and frame options using the `--debug` flag or going into interactive mode (`-i`) and then enter: `debug`. You can change the window and frame settings on the fly using, for exmaple: `w=2048` and `f=139`, respectively.


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


