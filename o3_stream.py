import requests
import io
import soundfile as sf
import sounddevice as sd

url = 'http://localhost:8000/tts/stream'

data = {
    'text': """Anthems were originally a form of liturgical music. In the Church of England, the rubric appoints them to follow the third collect at morning and evening prayer. Several anthems are included in the British coronation service.""",
    'voice': 'bm_fable',  # Change to the desired voice ID
    'speed': 1.0,
    'use_gpu': True,
    'format': 'wav'
}

# Send the POST request to the TTS server to stream the audio
response = requests.post(url, json=data, stream=True)

if response.status_code == 200:
    print('Streaming audio...')
    audio_buffer = io.BytesIO()
    
    # Read and stream the audio chunks as they come
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            audio_buffer.write(chunk)
    
    # Reset buffer position to the beginning
    audio_buffer.seek(0)

    # Read the audio data and samplerate using soundfile (works with multiple formats)
    data, samplerate = sf.read(audio_buffer)

    print(f'Playing the audio... (duration: {len(data) / samplerate:.2f}s)')
    
    # Play the audio in real-time
    sd.play(data, samplerate)
    sd.wait()  # Wait for the audio to finish playing
else:
    print(f'Error: Unable to fetch audio. Status code: {response.status_code}')

