import requests
import soundfile as sf
import io
import sounddevice as sd
import numpy as np

# URL for TTS service
url = 'http://localhost:8000/tts/stream'

# Long input text for TTS conversion
data = {
    'text': """The Renaissance period, spanning roughly from the 14th to the 17th century, was a time of profound cultural transformation in Europe. It marked the transition from the medieval era to the early modern period, characterized by a renewed interest in classical antiquity, humanism, and individualism. During this time, great advancements were made in art, science, philosophy, and literature, often attributed to the work of iconic figures such as Leonardo da Vinci, Michelangelo, Galileo Galilei, and William Shakespeare.""",
    'voice': 'bm_fable',  # Change the voice if needed
    'speed': 1.0,
    'use_gpu': True,
    'format': 'wav'
}

# Create a queue to hold audio chunks for real-time playback
audio_queue = []

# Callback function to play audio
def audio_callback(outdata, frames, time, status):
    # Fill the output buffer with audio data from the queue
    if len(audio_queue) > 0:
        chunk = audio_queue.pop(0)  # Get next chunk from the queue
        outdata[:len(chunk)] = chunk
    else:
        outdata.fill(0)  # Silence if no more data

# Send the request to the TTS service
response = requests.post(url, json=data, stream=True)

if response.status_code == 200:
    print('Streaming audio...')
    
    # Set up the sounddevice stream with the callback
    with sd.OutputStream(callback=audio_callback, channels=1, samplerate=16000):
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                # Convert chunk to numpy array
                data, samplerate = sf.read(io.BytesIO(chunk))
                # Append the chunk to the audio queue for playback
                audio_queue.append(data)
        
        sd.wait()  # Wait for the audio to finish playing
else:
    print(f'Error: Unable to fetch audio. Status code: {response.status_code}')

