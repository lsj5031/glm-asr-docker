# Real-time Transcription WebSocket API

## Endpoint
```
ws://host:18000/v1/audio/transcriptions/stream?language=auto
```

## Connection Flow
```
Client                              Server
  |                                    |
  |------ WebSocket Connect ---------->|
  |<----- Connection Accepted ---------|
  |                                    |
  |------ Audio chunk (binary) ------->|  (every 100-200ms)
  |         ...more chunks...          |
  |<----- {"text": "Hello", "final": false} --|  (on speech pause)
  |                                    |
  |------ Audio chunk (binary) ------->|
  |<----- {"text": "Hello world", "final": false} --|
  |                                    |
  |------ {"action": "stop"} --------->|  (user stops recording)
  |<----- {"text": "...", "final": true} --|
  |                                    |
  |------ Close ---------------------->|
```

## Client → Server Messages

| Type | Format | Description |
|------|--------|-------------|
| Audio | Binary | PCM 16-bit signed, 16kHz, mono. Send every 100-200ms |
| Stop | `{"action": "stop"}` | Signal end of recording |

## Audio Format Requirements
```
Format:     PCM (raw audio, no WAV header)
Bit depth:  16-bit signed integer (Int16)
Sample rate: 16000 Hz
Channels:   Mono (1 channel)
Byte order: Little-endian
Chunk size: ~3200 bytes = 100ms of audio
```

## Server → Client Messages

```json
{
  "text": "transcribed text so far",
  "final": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Cumulative transcription result |
| `final` | boolean | `true` when transcription is complete |
| `error` | string? | Optional error message if transcription failed |

## Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `language` | string | `"auto"` | Language hint (currently unused, reserved) |

## Transcription Triggers

The server uses Voice Activity Detection (VAD) to trigger transcription:
- **On speech pause**: Transcribes when 500ms of silence detected after speech
- **Buffer limit**: Forces transcription if buffer exceeds 30 seconds
- **On stop**: Final transcription when client sends stop signal

## TypeScript/JavaScript Example

```typescript
class RealtimeTranscriber {
  private ws: WebSocket | null = null;
  private audioContext: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private processor: ScriptProcessorNode | null = null;

  async start(onPartial: (text: string) => void, onFinal: (text: string) => void) {
    // Connect WebSocket
    this.ws = new WebSocket('ws://localhost:18000/v1/audio/transcriptions/stream');
    
    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.final) {
        onFinal(data.text);
      } else {
        onPartial(data.text);
      }
    };

    // Wait for connection
    await new Promise((resolve, reject) => {
      this.ws!.onopen = resolve;
      this.ws!.onerror = reject;
    });

    // Start audio capture
    this.mediaStream = await navigator.mediaDevices.getUserMedia({ 
      audio: { sampleRate: 16000, channelCount: 1 } 
    });
    
    this.audioContext = new AudioContext({ sampleRate: 16000 });
    const source = this.audioContext.createMediaStreamSource(this.mediaStream);
    
    // Process audio in chunks
    this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);
    this.processor.onaudioprocess = (e) => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        const float32 = e.inputBuffer.getChannelData(0);
        const int16 = new Int16Array(float32.length);
        for (let i = 0; i < float32.length; i++) {
          int16[i] = Math.max(-32768, Math.min(32767, float32[i] * 32768));
        }
        this.ws.send(int16.buffer);
      }
    };
    
    source.connect(this.processor);
    this.processor.connect(this.audioContext.destination);
  }

  stop() {
    // Send stop signal
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ action: 'stop' }));
    }
    
    // Cleanup
    this.processor?.disconnect();
    this.mediaStream?.getTracks().forEach(t => t.stop());
    this.audioContext?.close();
  }
}

// Usage
const transcriber = new RealtimeTranscriber();

// Start recording
await transcriber.start(
  (partial) => console.log('Partial:', partial),
  (final) => console.log('Final:', final)
);

// Stop recording (e.g., on button click)
transcriber.stop();
```

## C# Example

```csharp
using System.Net.WebSockets;

public class RealtimeTranscriber : IDisposable
{
    private ClientWebSocket _ws;
    private readonly Uri _uri = new("ws://localhost:18000/v1/audio/transcriptions/stream");
    
    public event Action<string, bool>? OnTranscription; // (text, isFinal)

    public async Task ConnectAsync(CancellationToken ct = default)
    {
        _ws = new ClientWebSocket();
        await _ws.ConnectAsync(_uri, ct);
        _ = ReceiveLoopAsync(ct);
    }

    public async Task SendAudioChunkAsync(byte[] pcm16Data, CancellationToken ct = default)
    {
        if (_ws.State == WebSocketState.Open)
        {
            await _ws.SendAsync(pcm16Data, WebSocketMessageType.Binary, true, ct);
        }
    }

    public async Task StopAsync(CancellationToken ct = default)
    {
        var stopMsg = Encoding.UTF8.GetBytes("{\"action\":\"stop\"}");
        await _ws.SendAsync(stopMsg, WebSocketMessageType.Text, true, ct);
    }

    private async Task ReceiveLoopAsync(CancellationToken ct)
    {
        var buffer = new byte[4096];
        while (_ws.State == WebSocketState.Open)
        {
            var result = await _ws.ReceiveAsync(buffer, ct);
            if (result.MessageType == WebSocketMessageType.Text)
            {
                var json = Encoding.UTF8.GetString(buffer, 0, result.Count);
                var msg = JsonSerializer.Deserialize<TranscriptionMessage>(json);
                OnTranscription?.Invoke(msg.Text, msg.Final);
            }
        }
    }

    public void Dispose() => _ws?.Dispose();
}

record TranscriptionMessage(string Text, bool Final);
```

## Error Handling

- If WebSocket disconnects unexpectedly, reconnect and resume
- If server returns `error` field, display to user and retry
- Implement exponential backoff for reconnection attempts

## Notes

- Audio is processed cumulatively (not chunked), so partial results may repeat/refine previous text
- VAD runs on every audio chunk, transcription only triggers on detected pauses
- For best results, ensure clean audio input (minimize background noise)
