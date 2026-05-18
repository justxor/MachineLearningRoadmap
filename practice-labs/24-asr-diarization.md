# 🎙️ Лаба 24: Распознавание речи (Whisper) + диаризация 🟡

## Цель

Построить пиплайн ASR + speaker diarization для расшифровки встреч и интервью. Научиться работать с аудио, выбирать размер модели и оценивать качество.

## Датасет

- AMI Meeting Corpus.
- Собранные свои записи (с согласия).
- LibriSpeech (простой ASR baseline).

## Минимальный пайплайн

1. Препроцессинг: ffmpeg к 16 kHz mono.
2. ASR: Whisper (large-v3) или faster-whisper.
3. Diarization: pyannote.audio (требует HF token).
4. Слияние: присвоить сегменты ASR спикерам.
5. Экспорт в SRT/JSON.
6. Post-processing: исправления через LLM (пунктуация, имена).

## Код: faster-whisper + pyannote

```python
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

asr = WhisperModel("large-v3", device="cuda", compute_type="float16")
segments, info = asr.transcribe("audio.wav", language="ru", vad_filter=True)

dia = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
d = dia("audio.wav")
for turn, _, speaker in d.itertracks(yield_label=True):
    print(f"{speaker} {turn.start:.1f}-{turn.end:.1f}")
```

## Метрики

- WER (Word Error Rate) — главная для ASR.
- DER (Diarization Error Rate).
- JER (Jaccard Error Rate) по спикерам.
- RTF (real-time factor).

## Расширения

- Punctuation + capitalization restoration.
- Translation через Whisper.
- Topic segmentation встреч (LLM-summary по блокам).
- Auto chapters / action items (LLM по transcript).
- Streaming ASR (вживую).

## Критерии приёмки

- [ ] WER <15% на русском (чистая запись).
- [ ] DER <15% при 2–3 спикерах.
- [ ] Поддержка форматов SRT, JSON, TXT.
- [ ] CLI / Gradio интерфейс.
- [ ] RTF <0.5 на GPU.
- [ ] Demo в репо с примером выхода.

## Анти-паттерны

- ❌ Обработка без VAD (в тишине Whisper «выдумывает»).
- ❌ Не нижать sample rate 16k — баги.
- ❌ Diarization на длинных файлах без chunking — OOM.
- ❌ Нет эвала («кажется круто»).

---

[← Назад к Practice Labs](./README.md)
