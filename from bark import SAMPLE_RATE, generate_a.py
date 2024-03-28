from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import scipy
import nltk
import numpy as np
from bark.generation import (
    generate_text_semantic,
    preload_models,
)
from bark.api import semantic_to_waveform
import os

os.environ["SUNO_USE_SMALL_MODELS"] = "True"
# download and load all models
preload_models()

script = """
Hi there [laughs] how are you
""".replace("\n", " ").strip()
sentences = nltk.sent_tokenize(script)

GEN_TEMP = 0.6
SPEAKER = "v2/en_speaker_9"
silence = np.zeros(int(0.15 * SAMPLE_RATE))  # quarter second of silence

pieces = []
for sentence in sentences:
   semantic_tokens = generate_text_semantic(
        sentence,
        history_prompt=SPEAKER,
        temp=GEN_TEMP,
        min_eos_p=0.05,  # this controls how likely the generation is to end
    )
   audio_array = semantic_to_waveform(semantic_tokens, history_prompt=SPEAKER,)
        #audio_array = generate_audio(sentence, history_prompt=SPEAKER)
   pieces += [audio_array, silence.copy()]


final_audio_array = np.concatenate(pieces)  # Assuming 'pieces' is your final audio data array
filename = "synth.wav"  # Name of the WAV file to create

  # Scale the data
scaled = np.int16(final_audio_array / np.max(np.abs(final_audio_array)) * 32767)

        # Write to the WAV file
scipy.io.wavfile.write("output.wav", rate=SAMPLE_RATE, data=scaled)
# Write the WAV file
#write_wav(filename, SAMPLE_RATE, audio_array)  # Ensure the data type is appropriate