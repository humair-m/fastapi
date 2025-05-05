python -c """
import requests, io, soundfile as sf, sounddevice as sd, numpy as np

def tts(text, voice):
    response = requests.post('http://localhost:8000/tts/stream', json={
        'text': text,
        'voice': voice,
        'speed': 1.0,
        'use_gpu': True,
        'format': 'wav'
    }, stream=True)
    if response.status_code != 200:
        raise RuntimeError(f'TTS error for {voice}: {response.status_code}')
    with io.BytesIO() as buf:
        for chunk in response.iter_content(1024):
            buf.write(chunk)
        buf.seek(0)
        audio, sr = sf.read(buf)
    return audio, sr

lines = [
    ('Fable will start with this opening line.', 'bm_fable'),
    ('Then Michael responds with his part.', 'am_michael'),
    ('Fable continues, rich and deep.', 'bm_fable'),
    ('Michael wraps it up with clarity.', 'am_michael')
]

audios = []
samplerate = None
for text, voice in lines:
    audio, sr = tts(text, voice)
    if samplerate is None:
        samplerate = sr
    elif sr != samplerate:
        raise ValueError('Sample rates don\'t match.')
    if audio.ndim > 1:  # convert to mono
        audio = np.mean(audio, axis=1)
    audios.append(audio)

# Mix with overlap
total_len = sum(len(a) for a in audios) + samplerate
mix = np.zeros(total_len)
pos = 0
for i, audio in enumerate(audios):
    start = max(0, pos - int(0.3 * samplerate)) if i > 0 else pos
    end = start + len(audio)
    mix[start:end] += audio
    pos = start + len(audio)

# Add soft background sine wave (mono-safe)
bg = 0.03 * np.sin(2 * np.pi * 220 * np.arange(len(mix)) / samplerate)
mix += bg

# Normalize
mix /= np.max(np.abs(mix))
print(f'Playing overlapping, alternating voices with background music... ({len(mix)/samplerate:.2f}s)')
sd.play(mix, samplerate)
sd.wait()
"""

