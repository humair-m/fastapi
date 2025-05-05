import requests
import soundfile as sf
import io
import sounddevice as sd

# URL for TTS service
url = 'http://localhost:8000/tts/stream'

# Long input text for TTS conversion
data = {
    'text': """The Renaissance period, spanning roughly from the 14th to the 17th century, was a time of profound cultural transformation in Europe. It marked the transition from the medieval era to the early modern period, characterized by a renewed interest in classical antiquity, humanism, and individualism. During this time, great advancements were made in art, science, philosophy, and literature, often attributed to the work of iconic figures such as Leonardo da Vinci, Michelangelo, Galileo Galilei, and William Shakespeare.
    
    In the visual arts, the Renaissance is perhaps most famous for its dramatic evolution of techniques and the human figure, with artists like Raphael and Titian redefining perspectives and anatomy. The use of light and shadow (chiaroscuro) became a hallmark of the period, adding depth and realism to paintings. Sculpture also flourished, with Michelangelo’s "David" and Donatello’s works showcasing the period's emphasis on naturalism and the beauty of the human form.
    
    In literature, the Renaissance produced some of the greatest works of Western civilization. Writers like Dante Alighieri, Geoffrey Chaucer, and John Milton created masterpieces that continue to shape literary tradition. The period’s exploration of human emotions, individual agency, and the complexities of the human soul offered new narrative possibilities, expanding beyond the religious themes of the Middle Ages.
    
    The Renaissance was also a time of great scientific discovery. Figures like Copernicus, Kepler, and Galileo revolutionized our understanding of the cosmos, challenging the geocentric model of the universe that had dominated for centuries. This shift in thinking paved the way for the Scientific Revolution, which would have far-reaching consequences for future generations.
    
    The political landscape of the Renaissance was equally dynamic. The rise of powerful city-states in Italy, such as Florence, Venice, and Milan, provided fertile ground for the flourishing of arts and culture. The Medici family in Florence, for example, were influential patrons of the arts, funding the work of renowned artists and thinkers. The period also saw the gradual emergence of nation-states in Europe, with monarchs asserting greater control over their territories and military power.""",
    'voice': 'bm_fable',  # Change the voice if needed
    'speed': 1.0,
    'use_gpu': True,
    'format': 'wav'
}

# Send the request to the TTS service
response = requests.post(url, json=data, stream=True)

if response.status_code == 200:
    print('Streaming audio...')
    with io.BytesIO() as audio_buffer:
        # Write the streamed audio data to buffer
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                audio_buffer.write(chunk)
        audio_buffer.seek(0)
        
        # Read the audio data into an array using soundfile
        data, samplerate = sf.read(audio_buffer)
        
        # Optionally save the audio to a file
        sf.write('output_audio.wav', data, samplerate)
        
        # Play the audio using sounddevice
        print(f'Playing audio... (duration: {len(data)/samplerate:.2f} seconds)')
        sd.play(data, samplerate)
        sd.wait()  # Wait for the audio to finish playing
else:
    print(f'Error: Unable to fetch audio. Status code: {response.status_code}')

