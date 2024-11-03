from resemblyzer import VoiceEncoder
from resemblyzer.audio import preprocess_wav

def get_content_vector(wav_fpath):
    """
    Given a path to a .wav file, return its content vector.
    """
    wav = preprocess_wav(wav_fpath)
    encoder = VoiceEncoder()
    embed = encoder.embed_utterance(wav)
    return embed

output = get_content_vector("neutral_1-28_0001.wav")
print(output)
print(output.shape)
print(type(output))
