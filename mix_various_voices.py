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

a1, sr1 = tts('This is spoken by a British male, known as Fable.', 'bm_fable')
a2, sr2 = tts('And now this is Michael, an American male voice.', 'am_michael')

if sr1 != sr2:
    raise ValueError('Sample rates do not match.')

final = np.concatenate((a1, a2))
print(f'Playing mixed voices... (duration: {len(final)/sr1:.2f} seconds)')
sd.play(final, sr1)
sd.wait()
"""

