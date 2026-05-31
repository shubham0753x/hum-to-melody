## input audio
import sounddevice as sd
from scipy.io.wavfile import write
import os

intermediates_folder = 'intermediate'
os.makedirs(intermediates_folder, exist_ok=True)

generated_folder = 'generated'
os.makedirs(generated_folder, exist_ok=True)

def record(duration = 10,sample_rate = 44100):
    i = 0
    while os.path.exists(f'whistle{i}.wav'):
        i += 1

    filename = f'whistle{i}.wav'
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    print("Done")
    write(filename, sample_rate, audio)
    return filename

## clean audio
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

def clean_audio(filename: str):
    audio = AudioSegment.from_file(filename)
    non_silent = detect_nonsilent(audio, silence_thresh=-40)
    if non_silent:
        start, end = non_silent[0][0], non_silent[-1][1]
        trimmed = audio[start:end]
    else:
        trimmed = audio
    out_name = os.path.basename(filename)[:-4] + 'clean.wav'
    saving_path = os.path.join(intermediates_folder,out_name)
    trimmed.export(saving_path, format='wav')
    return saving_path

## .wav to midi
from basic_pitch.inference import predict

def wav_to_midi(file :str):
    model_output, midi_data, note_events = predict(file)
    # save the midi
    output_name = os.path.basename(file).replace('.wav', '.midi')
    saving_path = os.path.join(intermediates_folder, output_name)
    midi_data.write(saving_path)
    return saving_path



##convert clean midi to abc
import subprocess
import pickle as pkl

def midi_to_abc(file :str):
    out_name = os.path.basename(file).replace('.midi', '.abc')
    saving_path = os.path.join(intermediates_folder,out_name)
    subprocess.run(['midi2abc', file, '-o', saving_path])
    return saving_path

## preprocess whistle_abc
def preprocess_abc(file : str):
    if '.abc' not in file:
        raise ValueError("Expected abc file")
    with open(file,'r') as f:
        tune = f.read()
    tune = tune.split('\n')
    tune = [line for line in tune if (len(line)>0 and line[0] not 
                     in ('S', 'T','V', '%', 'X', 'P', 'Y', 'H', 'N', 'Z','\n'))]
    tune = [line.split('%')[0] for line in tune]
    print(tune)
    tune = '\n'.join(tune)
    tune = '`'+tune
    with open('stoi.pkl', 'rb') as f:
        stoi = pkl.load(f)
    tune = ''.join([c for c in tune if c in stoi])
    return tune


## inference
import pickle as pkl
import torch
from music21 import converter
import uuid
import time


def generate_tone(initial_abc,Model,model_weights,itos_pkl,stoi_pkl,max_len = 1000,max_attempt = 10):
    with open(itos_pkl,'rb') as f:
        itos = pkl.load(f)

    with open(stoi_pkl,'rb') as f:
        stoi = pkl.load(f)


    Model.load_state_dict(
        torch.load(model_weights, weights_only=False,map_location=torch.device('cpu'))
    )

    Model.eval()

    hidden = None
    with torch.no_grad():
        for X in initial_abc[:-1]:
            X = stoi[X]
            X = torch.tensor([[X]])
            _,hidden = Model(X,hidden)

    def infer():
        output = initial_abc[1:]
        X = initial_abc[-1]
        X = stoi[X]
        hidden1 = hidden
        for _ in range(max_len):
            X = torch.tensor([[X]])
            logit,hidden1 = Model(X,hidden1)
            prob = torch.softmax(logit.squeeze() , dim=-1)
            next_token = torch.multinomial(prob, num_samples=1).item()
            if next_token == stoi['$']:
                break
            output += itos[next_token]
            X = next_token
    
        return output
    
   
    out_file_name = f'{uuid.uuid4()}.mid'

    for attempt in range(max_attempt):
        tune = infer()
        try:
            score = converter.parse(tune, format='abc')
            saving_path = os.path.join(intermediates_folder,out_file_name)
            score.write('midi', saving_path)
            print(f'Success on attempt {attempt+1}')
            break
        except:
            print(f'Attempt {attempt+1} failed, retrying...')

    return saving_path


def generatedmidi_to_wav(generated_file: str):
    out_name = os.path.basename(generated_file).replace('.mid', '.wav')
    saving_path = os.path.join(generated_folder, out_name)
    subprocess.run([
        'fluidsynth', '-ni', '-T', 'wav',
        '-o', f'audio.file.name={saving_path}',
        '-o', 'audio.driver=file',
        '-o', 'synth.gain=2.0',
        '/usr/share/soundfonts/FluidR3_GM.sf2',
        generated_file
    ])
    prev_size = -1
    while True:
        curr_size = os.path.getsize(saving_path) if os.path.exists(saving_path) else 0
        if curr_size == prev_size and curr_size > 0:
            break
        prev_size = curr_size
        time.sleep(0.5)
        
    return saving_path
    
