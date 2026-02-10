# Quick Start: Global GLM-ASR CLI

## Install (One-Time Setup)

```bash
# Option 1: Symlink (easiest)
sudo ln -s /home/leo/code/glm-asr/glm-asr /usr/local/bin/glm-asr

# Option 2: Add to PATH
echo 'export PATH="/home/leo/code/glm-asr:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Option 3: Copy to bin
mkdir -p ~/.local/bin
cp /home/leo/code/glm-asr/glm-asr ~/.local/bin/
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

## Use from Anywhere

```bash
# Transcribe any file on your system
glm-asr transcribe ~/Downloads/podcast.mp3
glm-asr transcribe ~/Videos/lecture.mp4
glm-asr transcribe ~/Music/song.wav

# With options
glm-asr transcribe audio.mp3 -o output.txt
glm-asr transcribe video.mp4 -l zh --format srt
glm-asr transcribe meeting.m4a -l en -c 10

# Check status
glm-asr status
glm-asr health
glm-asr start
glm-asr stop
```

## What Happens

1. You run: `glm-asr transcribe ~/audio.mp3`
2. Script copies file into container
3. Container transcribes the file
4. Script copies transcript back to ~/audio.txt
5. Done! ✅

## Requirements

- Container must be running: `glm-asr start`
- File must exist: `ls ~/audio.mp3`
- Enough disk space for transcript

## Troubleshooting

```bash
# Container not running?
glm-asr start

# Command not found?
which glm-asr
# Should show: /usr/local/bin/glm-asr or similar

# Permission denied?
chmod +x /home/leo/code/glm-asr/glm-asr
```

## Examples

```bash
# Podcast to text
glm-asr transcribe ~/Downloads/podcast.mp3

# Video to SRT subtitles
glm-asr transcribe ~/Videos/lecture.mp4 --format srt

# Chinese audio
glm-asr transcribe ~/audio/speech.wav -l zh

# Custom output location
glm-asr transcribe input.mp3 -o ~/Documents/transcript.txt

# Batch process (use find)
find ~/Music -name "*.mp3" -exec glm-asr transcribe {} \;
```

## Tips

- Output file is saved next to input file
- Use absolute paths for reliability
- Check status before transcribing: `glm-asr status`
- Keep container running for multiple files
- All CLI options work the same as local usage

## Comparison

### Before (Local CLI)
```bash
cd ~/code/glm-asr
cp ~/Downloads/audio.mp3 data/
make transcribe INPUT=/app/data/audio.mp3
```

### After (Global CLI)
```bash
glm-asr transcribe ~/Downloads/audio.mp3
```

Much simpler! 🎉
