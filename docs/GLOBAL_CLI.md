# Using GLM-ASR CLI from Anywhere

## Quick Setup

### Option 1: Add to PATH (Recommended)

```bash
# 1. Create a symlink in your PATH
sudo ln -s /home/leo/code/glm-asr/glm-asr /usr/local/bin/glm-asr

# 2. Or add to your ~/.bashrc or ~/.zshrc
echo 'export PATH="/home/leo/code/glm-asr:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 3. Now you can use it from anywhere!
glm-asr transcribe ~/Downloads/audio.mp3
```

### Option 2: Install to Home Directory

```bash
# 1. Create bin directory
mkdir -p ~/.local/bin

# 2. Copy the script
cp /home/leo/code/glm-asr/glm-asr ~/.local/bin/

# 3. Add to PATH (if not already there)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 4. Use from anywhere
glm-asr transcribe ~/Music/podcast.mp3
```

### Option 3: System-Wide Installation

```bash
# 1. Copy to /usr/local/bin
sudo cp /home/leo/code/glm-asr/glm-asr /usr/local/bin/

# 2. Make executable
sudo chmod +x /usr/local/bin/glm-asr

# 3. Use from anywhere
glm-asr transcribe ~/Videos/lecture.mp4
```

## Usage Examples

### Basic Usage

```bash
# Transcribe from anywhere
glm-asr transcribe ~/Downloads/podcast.mp3

# With custom output
glm-asr transcribe ~/audio.wav -o ~/Documents/transcript.txt

# With language
glm-asr transcribe ~/speech.mp3 -l en

# To SRT format
glm-asr transcribe ~/video.mp4 --format srt

# All CLI options work
glm-asr transcribe ~/audio.m4a -l zh -f srt -c 10
```

### Container Management

```bash
# Check if container is running
glm-asr status

# Start container
glm-asr start

# Stop container
glm-asr stop

# Check health
glm-asr health
```

## How It Works

The global CLI wrapper:

1. **Accepts any file path** on your system
2. **Copies the file** into the container temporarily
3. **Runs the transcription** using the internal CLI
4. **Copies the result back** to the original file's directory
5. **Cleans up** temporary files

### Example Flow

```bash
$ glm-asr transcribe ~/Downloads/podcast.mp3

GLM-ASR Transcription
Input: ~/Downloads/podcast.mp3
Size: 45.2M
Copying file to container...
Transcribing: 100%|██████████| 2/2 [00:25<00:00, 12.3s/chunk]
Copying transcript back...
✓ Transcription complete!
Output: ~/Downloads/podcast.txt
```

## Configuration

### Set Custom Project Directory

If your GLM-ASR project is not in the default location:

```bash
# Temporary
export GLM_ASR_HOME=/path/to/glm-asr
glm-asr transcribe audio.mp3

# Permanent (add to ~/.bashrc)
echo 'export GLM_ASR_HOME=/path/to/glm-asr' >> ~/.bashrc
```

### Set Custom Container Name

```bash
# Temporary
export GLM_ASR_CONTAINER=my-glm-asr
glm-asr transcribe audio.mp3

# Permanent
echo 'export GLM_ASR_CONTAINER=my-glm-asr' >> ~/.bashrc
```

## Limitations

1. **Container must be running**: The CLI requires the Docker container to be running
2. **One container at a time**: Uses the default container name (glm-asr)
3. **File copying**: Large files may take time to copy into/out of container
4. **Network only**: Cannot work offline (communicates with container via docker exec)

## Advantages

1. **Use from anywhere**: No need to cd into project directory
2. **Automatic file handling**: Copies files to/from container
3. **Same CLI interface**: All options work the same
4. **Clean output**: Transcript saved next to original file
5. **Easy management**: Start/stop/status commands

## Comparison

### Local Project CLI (Original)
```bash
cd /home/leo/code/glm-asr
cp ~/Downloads/audio.mp3 data/
make transcribe INPUT=/app/data/audio.mp3
```

### Global CLI (New)
```bash
glm-asr transcribe ~/Downloads/audio.mp3
```

## Troubleshooting

### "Container not running"
```bash
# Start the container
glm-asr start

# Or manually
cd /home/leo/code/glm-asr
docker compose up -d
```

### "Command not found"
```bash
# Check if script is in PATH
which glm-asr

# If not, add to PATH
export PATH="/home/leo/code/glm-asr:$PATH"

# Or create symlink
sudo ln -s /home/leo/code/glm-asr/glm-asr /usr/local/bin/glm-asr
```

### "Permission denied"
```bash
# Make script executable
chmod +x /home/leo/code/glm-asr/glm-asr
```

### File not found
```bash
# Use absolute path
glm-asr transcribe /full/path/to/audio.mp3

# Or relative path from current directory
glm-asr transcribe ./audio.mp3
```

## Shell Integration

### Add to ~/.bashrc or ~/.zshrc

```bash
# GLM-ASR Global CLI
export PATH="/home/leo/code/glm-asr:$PATH"
export GLM_ASR_HOME="/home/leo/code/glm-asr"

# Optional: Aliases for convenience
alias transcribe='glm-asr transcribe'
alias glm-status='glm-asr status'
alias glm-health='glm-asr health'
```

### Then use aliases

```bash
transcribe ~/audio.mp3
glm-status
glm-health
```

## Examples by Use Case

### Transcribe Podcast
```bash
glm-asr transcribe ~/Downloads/podcast.mp3 -o ~/Documents/podcasts/transcript.txt
```

### Transcribe Lecture Video
```bash
glm-asr transcribe ~/Downloads/lecture.mp4 --format srt
```

### Transcribe Meeting Recording
```bash
glm-asr transcribe ~/Meetings/2025-02-10.m4a -l en
```

### Batch Transcribe (use find)
```bash
find ~/Music/Podcasts -name "*.mp3" -exec glm-asr transcribe {} \;
```

## Performance Tips

1. **Start container once**: Keep it running for multiple transcriptions
2. **Use absolute paths**: Faster than relative paths
3. **Check status first**: `glm-asr status` before transcribing
4. **Monitor with logs**: `cd ~/code/glm-asr && docker compose logs -f`

## Next Steps

After setting up the global CLI:

1. Add to your PATH permanently
2. Create convenient aliases
3. Set up environment variables in your shell config
4. Test with a sample file
5. Enjoy transcribing from anywhere!
