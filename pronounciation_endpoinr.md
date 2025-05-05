# 📘 Kokoro TTS API: Pronunciation Endpoint

The Kokoro TTS API supports custom pronunciation overrides, allowing users to control how specific words are spoken during synthesis. This is particularly useful for names, foreign words, technical terms, or alternative phonetics.

---

## 🔤 Add a Custom Pronunciation

### Endpoint
```
POST /pronunciation
```

### Parameters (Query)
| Name           | Type   | Description                                      |
|----------------|--------|--------------------------------------------------|
| `word`         | string | The word to override                             |
| `pronunciation`| string | The phonetic pronunciation to use                |
| `language_code`| string | The language code (e.g. `a`, `b`, etc.)          |

### Example
```bash
curl -X POST "http://localhost:8000/pronunciation?word=kokoro&pronunciation=koh-koh-roh&language_code=a"
```

---

## 🗣️ Use Custom Pronunciation in TTS

To use your custom pronunciation in a synthesis request:

### Endpoint
```
POST /tts or /tts/stream
```

### JSON Body Example
```json
{
  "text": "Welcome to Kokoro, a modern TTS model.",
  "voice": "af_heart",
  "speed": 1.0,
  "use_gpu": true,
  "format": "wav",
  "pronunciations": {
    "kokoro": "koh-koh-roh"
  }
}
```

---

## 📃 List Pronunciations

### Endpoint
```
GET /pronunciations?language_code={code}
```

### Example
```bash
curl "http://localhost:8000/pronunciations?language_code=a"
```

---

## ❌ Delete a Pronunciation

### Endpoint
```
DELETE /pronunciations/{word}?language_code={code}
```

### Example
```bash
curl -X DELETE "http://localhost:8000/pronunciations/kokoro?language_code=a"
```

---

## Notes
- Pronunciations are case-sensitive.
- Use simple syllable spacing (e.g., "mi-kro-bi-ol-o-gy") for clarity.
- The `language_code` must match the voice’s category.

---

For best results, always test with `/tts/stream` to hear the immediate output of your pronunciation settings.
