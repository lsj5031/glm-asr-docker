# Supported Audio/Video Formats

## Overview

GLM-ASR supports **virtually all audio and video formats** thanks to **ffmpeg** integration via **pydub**. The server automatically converts any input format to the optimal format for the model.

## Audio Formats

### Common Formats
- ✅ **MP3** - MPEG Audio Layer III
- ✅ **WAV** - Waveform Audio File Format
- ✅ **FLAC** - Free Lossless Audio Codec
- ✅ **M4A** - MPEG-4 Audio
- ✅ **AAC** - Advanced Audio Coding
- ✅ **OGG** - Ogg Vorbis/Opus
- ✅ **WMA** - Windows Media Audio

### Additional Formats
- ✅ **OPUS** - Opus Codec
- ✅ **AIFF/AIF** - Audio Interchange File Format
- ✅ **AU** - Sun Microsystems Audio
- ✅ **RA** - RealAudio
- ✅ **AMR** - Adaptive Multi-Rate
- ✅ **DSD** - Direct Stream Digital
- ✅ **APE** - Monkey's Audio
- ✅ **WV** - WavPack
- ✅ **TTA** - The True Audio
- ✅ **AC3** - Dolby Digital
- ✅ **DTS** - Digital Theater Systems

## Video Formats (Audio Track Extraction)

The server automatically extracts the audio track from video files:

### Common Video Formats
- ✅ **MP4** - MPEG-4 Part 14
- ✅ **MOV** - QuickTime File Format
- ✅ **AVI** - Audio Video Interleave
- ✅ **MKV** - Matroska Video
- ✅ **WEBM** - Web Media Project
- ✅ **WMV** - Windows Media Video
- ✅ **FLV** - Flash Video
- ✅ **3GP** - Third Generation Partnership Project

### Additional Video Formats
- ✅ **MPG/MPEG** - MPEG-1/MPEG-2
- ✅ **M4V** - MPEG-4 Video
- ✅ **TS** - MPEG Transport Stream
- ✅ **MTS** - AVCHD Video
- ✅ **VOB** - DVD Video Object
- ✅ **RM** - RealMedia
- ✅ **RMVB** - RealMedia Variable Bitrate
- ✅ **ASF** - Advanced Systems Format

## Container Formats

- ✅ **Matroska (.mkv)**
- ✅ **MPEG-4 (.mp4, .m4a, .m4v)**
- ✅ **MPEG-PS (.mpg, .mpeg)**
- ✅ **MPEG-TS (.ts)**
- ✅ **AVI (.avi)**
- ✅ **Wave (.wav)**
- ✅ **Ogg (.ogg, .oga)**
- ✅ **WebM (.webm)**
- ✅ **FLV (.flv)**
- ✅ **3GPP (.3gp)**
- ✅ **QuickTime (.mov)**
- ✅ **RealMedia (.rm, .rmvb)**

## How It Works

1. **Upload** - Any supported format is accepted
2. **Detection** - Server detects format and properties
3. **Conversion** - ffmpeg converts to WAV (16kHz, mono, PCM)
4. **Processing** - Audio is transcribed by GLM-ASR model
5. **Output** - Transcript returned in requested format (text/SRT)

## Examples

```bash
# Audio files
make transcribe INPUT=/app/data/podcast.mp3
make transcribe INPUT=/app/data/interview.wav
make transcribe INPUT=/app/data/meeting.m4a
make transcribe INPUT=/app/data/song.flac

# Video files (audio is automatically extracted)
make transcribe INPUT=/app/data/lecture.mp4
make transcribe INPUT=/app/data/presentation.mov
make transcribe INPUT=/app/data/interview.avi
make transcribe INPUT=/app/data/webinar.mkv

# Batch process mixed formats
cp ~/media/*.{mp3,wav,m4a,mp4,mov} data/
make transcribe INPUT=/app/data/
```

## Format-Specific Notes

### MP3
- Most common format
- Good compression ratio
- Universal compatibility

### WAV
- Uncompressed audio
- Larger file size
- Best quality (though server resamples anyway)

### M4A
- Apple's default format
- Better compression than MP3
- Widely supported

### FLAC
- Lossless compression
- Larger file size
- No quality loss (though server resamples anyway)

### MP4/MOV/AVI (Video)
- Server extracts audio track automatically
- All codecs supported (H.264, H.265, VP9, etc.)
- Subtitle tracks are ignored (use --format srt to generate)

## Quality Considerations

The server automatically:
- Resamples to **16kHz** (optimal for the model)
- Converts to **mono** (single channel)
- Uses **16-bit PCM** format
- Applies **VAD** (Voice Activity Detection) for chunking

Therefore, the input format doesn't affect transcription quality - the server normalizes everything to the optimal format for the model.

## Troubleshooting

### File Not Supported
If you get a "format not supported" error:
1. Ensure ffmpeg can decode the file: `ffmpeg -i yourfile.ext`
2. Check the file isn't corrupted
3. Try converting to a common format (MP3/WAV)

### Video File Issues
- Server only transcribes the audio track
- Ensure the video has an audio track
- Some DRM-protected files may not work

### Large Files
- Maximum size: 500MB (configurable via MAX_AUDIO_SIZE_MB)
- Long files are automatically chunked
- Use `-c` flag to adjust chunk size

## Adding New Formats

To add support for additional formats:
1. Ensure ffmpeg supports the format
2. No code changes needed - pydub handles it automatically
3. Test with: `docker compose exec glm-asr python glm_asr_cli.py transcribe /app/data/yourfile.ext`

## Performance by Format

All formats have similar transcription performance because:
- Input format doesn't affect processing speed
- Server converts everything to the same internal format
- Only difference is upload/conversion time (usually <1 second)

Typical conversion times:
- MP3/M4A: <0.5s per minute of audio
- WAV: No conversion needed (instant)
- FLAC: <1s per minute of audio
- Video files: 1-2s per minute (for audio extraction)
