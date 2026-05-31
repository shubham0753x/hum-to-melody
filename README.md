# hum-to-melody

Hum a tune into your browser and let a neural network continue it as a melody.

## How it works

The pipeline has two parts:

**Input processing**
1. Browser records your humming via microphone
2. [Basic Pitch](https://github.com/spotify/basic-pitch) (Spotify) transcribes the audio to MIDI
3. `midi2abc` converts MIDI to ABC notation
4. The ABC is cleaned and fed as a seed to the model

**Generation**
5. A character-level LSTM trained on the [Nottingham Music Database](https://abc.sourceforge.net/NMD/) continues the ABC sequence
6. The generated ABC is converted back to MIDI via `music21`
7. FluidSynth renders the MIDI to WAV using the FluidR3 soundfont
8. The browser plays the result

The model never sees raw audio — it operates entirely on symbolic music notation (ABC), the same format a musician would read and play from.

## Architecture

- **Model**: 2-layer LSTM, hidden size 256, character-level tokenization over ABC notation
- **Vocabulary**: 62 characters (notes, durations, barlines, header fields)
- **Training data**: ~1000 folk tunes from the Nottingham Music Database
- **Training**: CrossEntropyLoss, Adam optimizer, ~25 epochs on GPU

## Stack

- **ML**: PyTorch, Basic Pitch (TensorFlow)
- **Audio**: FluidSynth, abcmidi, pydub
- **Backend**: Flask
- **Frontend**: Vanilla HTML/CSS/JS, MediaRecorder API

## Local Setup

**System dependencies**
```bash
sudo pacman -S abcmidi fluidsynth soundfont-fluid   # Arch Linux
# or
sudo apt install abcmidi fluidsynth fluid-soundfont-gm  # Ubuntu/Debian
```

**Python dependencies**
```bash
pip install -r requirements.txt
```

**Model weights and pkl files**

Download from [releases](https://github.com/shubham0753x/hum-to-melody/releases) and place in the project root:
- `music_lstm_best.pth`
- `stoi.pkl`
- `itos.pkl`

**Run**
```bash
python app.py
```

Open `http://localhost:5000`, click Record, hum a melody, click Stop and wait.

## Training

To retrain the model from scratch, run the preprocessing notebook first:

```bash
jupyter notebook data_preprocessing.ipynb
```

Then train:

```bash
python train.py
```

## Limitations

- Model trained on Irish/English folk melodies — generated output reflects that style
- Generation takes ~30 seconds (FluidSynth rendering is real-time)
- Input quality matters — hum clearly in a quiet environment for best transcription

## Live Demo

- Deployed the project on huggingface spaces using Docker.
- Try it [here](https://huggingface.co/spaces/shubham0753x/hum-to-melody)
