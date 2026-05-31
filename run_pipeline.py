import utility
import pickle as pkl
from train import MusicLSTM
def local_run():
    recording = utility.record() 
    cleaned_audio = utility.clean_audio(recording)
    whistle_midi = utility.wav_to_midi(cleaned_audio)
    whistle_abc = utility.midi_to_abc(whistle_midi)
    cleaned_abc = utility.preprocess_abc(whistle_abc)

    with open('itos.pkl','rb') as f:
        vocab = pkl.load(f)
        vocab_size = len(vocab)

    Model = MusicLSTM(vocab_size,64,256,2,0.4)

    generated_midi = utility.generate_tone(cleaned_abc,Model,"music_lstm_best.pth",'itos.pkl','stoi.pkl')

    generated_wav = utility.generatedmidi_to_wav(generated_midi)

    print(f'{generated_wav} saved')

def online_run(input_path):
    cleaned_audio = utility.clean_audio(input_path)
    whistle_midi = utility.wav_to_midi(cleaned_audio)
    whistle_abc = utility.midi_to_abc(whistle_midi)
    cleaned_abc = utility.preprocess_abc(whistle_abc)

    with open('itos.pkl','rb') as f:
        vocab = pkl.load(f)
        vocab_size = len(vocab)

    Model = MusicLSTM(vocab_size,64,256,2,0.4)

    generated_midi = utility.generate_tone(cleaned_abc,Model,"music_lstm_best.pth",'itos.pkl','stoi.pkl')

    generated_wav = utility.generatedmidi_to_wav(generated_midi)

    print(f'{generated_wav} saved')
    return generated_wav
    


if __name__ == '__main__':
    local_run()