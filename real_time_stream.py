python -c """
import requests
import soundfile as sf
import io
import sounddevice as sd

url = 'http://localhost:8000/tts/stream'

data = {
    'text': 'Anthems were originally a form of liturgical music. In the Church of England, the rubric appoints them to follow the third collect at morning and evening prayer. Several anthems are included in the British coronation service. The words are selected from Holy Scripture or in some cases from the Liturgy and the music is generally more elaborate and varied than that of psalm or hymn tunes. Being written for a trained choir rather than the congregation, the Anglican anthem is analogous to the motet of the Catholic and Lutheran Churches but represents an essentially English musical form. Anthems may be described as \\\"verse\\\", \\\"full\\\", or \\\"full with verse\\\", depending on whether they are intended for soloists, the full choir, or both. Another way of describing an anthem is that it is a piece of music written specifically to fit a certain accompanying text, and it is often difficult to make any other text fit that same melodic arrangement. It also often changes melody and/or meter, frequently multiple times within a single song, and is sung straight through from start to finish, without repeating the melody for following verses like a normal song (although certain sections may be repeated when marked). An example of an anthem with multiple meter shifts, fuguing, and repeated sections is Claremont, or Vital Spark of Heav\\'nly Flame. Another well known example is William Billing\\'s Easter Anthem, also known as \\\"The Lord Is Risen Indeed!\\\" after the opening lines. This anthem is still one of the more popular songs in the Sacred Harp tune book. The anthem developed as a replacement for the Catholic votive antiphon commonly sung as an appendix to the main office to the Blessed Virgin Mary or other saints.',
    'voice': 'em_alex',
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
        sd.wait()
else:
    print(f'Error: Unable to fetch audio. Status code: {response.status_code}')
"""

