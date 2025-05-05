python -c """
import requests
import soundfile as sf
import io
import sounddevice as sd

url = 'http://localhost:8000/tts/stream'

data = {
    'text': 'Real-time text-to-speech is now speaking as it generates audio.',
    'voice': 'am_michael',
    'speed': 1.0,
    'use_gpu': True,
    'format': 'wav'
}

response = requests.post(url, json=data, stream=True)

if response.status_code == 200:
    print('Streaming audio...')
    with io.BytesIO() as audio_buffer:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                audio_buffer.write(chunk)
        audio_buffer.seek(0)
        data, samplerate = sf.read(audio_buffer)
        sf.write('temp_audio.wav', data, samplerate)
        print(f'Playing the audio... (duration: {len(data)/samplerate:.2f} seconds)')
        sd.play(data, samplerate)
        sd.wait()  # Wait for the audio to finish playing
else:
    print(f'Error: Unable to fetch audio. Status code: {response.status_code}')
"""

