from flask import Flask, request, jsonify
import asyncio
import aiohttp
from aiohttp import ClientTimeout
import discord
import io
import logging
from scipy.io.wavfile import write as write_wav
import scipy.io.wavfile
import numpy as np
import os
import nltk

# Ensure all necessary imports for Bark and its dependencies are included
from bark import SAMPLE_RATE, generate_audio, preload_models
from bark.generation import generate_text_semantic
from bark.api import semantic_to_waveform

# Set the environment variable for using smaller models and preload models
os.environ["SUNO_USE_SMALL_MODELS"] = "True"
preload_models()

app = Flask(__name__)

async def generate_audio_and_send_to_discord(text, webhook_url):
    # Generate audio from text
    audio_data, sample_rate = generate_audio_function(text)
    if audio_data is None:
        return "Failed to generate audio", 500

    # Convert numpy array to bytes for sending
    audio_bytes = io.BytesIO()
    scipy.io.wavfile.write(audio_bytes, sample_rate, audio_data)
    audio_bytes.seek(0)  # Rewind to the start

    # Use aiohttp to send the generated audio to Discord
    async with aiohttp.ClientSession() as session:
        webhook = discord.Webhook.from_url(webhook_url, session=session)
        await webhook.send(username="Synth Bot", file=discord.File(fp=audio_bytes, filename="tts_output.wav"))
    
    return "Audio message sent to Discord", 200

def generate_audio_function(text):
    sentences = nltk.sent_tokenize(text)

    GEN_TEMP = 0.6
    SPEAKER = "v2/en_speaker_9"
    silence = np.zeros(int(0.15 * SAMPLE_RATE))  # quarter second of silence

    pieces = []
    for sentence in sentences:
        semantic_tokens = generate_text_semantic(
            sentence,
            history_prompt=SPEAKER,
            temp=GEN_TEMP,
            min_eos_p=0.05,  # Controls how likely the generation is to end
        )
        audio_array = semantic_to_waveform(semantic_tokens, history_prompt=SPEAKER)
        pieces += [audio_array, silence.copy()]

    final_audio_array = np.concatenate(pieces)

    # Scale the data
    scaled = np.int16(final_audio_array / np.max(np.abs(final_audio_array)) * 32767)
    
    return scaled, SAMPLE_RATE

@app.route('/synthesize_and_send', methods=['POST'])
def synthesize_and_send():
    data = request.json
    text = data.get('text', '').strip()
    webhook_url = data.get('webhook_url')

    if not text or not webhook_url:
        return jsonify({"error": "Missing text or webhook_url"}), 400

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result, status = loop.run_until_complete(generate_audio_and_send_to_discord(text, webhook_url))

    return jsonify({"message": result}), status

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
