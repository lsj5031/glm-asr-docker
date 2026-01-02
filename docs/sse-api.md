# SSE Streaming Transcription API

## Endpoint
`POST /v1/audio/transcriptions`

## Request

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | Yes | Audio file (WAV, MP3, M4A, etc.) |
| `language` | string | No | Language hint (default: `"auto"`) |
| `stream` | boolean | No | Enable SSE streaming (default: `false`) |

## Response (Non-Streaming: `stream=false`)

**Content-Type:** `application/json`

```json
{
  "text": "Full transcription text here"
}
```

## Response (Streaming: `stream=true`)

**Content-Type:** `text/event-stream`

Server sends SSE events as chunks are transcribed:

```
data: First chunk transcription text

data: Second chunk transcription text

data: [DONE]

```

**Event Format:**
- Each event is prefixed with `data: ` followed by transcription text
- Events are separated by double newlines (`\n\n`)
- Final event is `data: [DONE]` signaling stream completion
- Error events: `data: [Error: <message>]`

## C# Client Example

```csharp
public async IAsyncEnumerable<string> TranscribeWithStreamingResponseAsync(
    string audioFilePath,
    [EnumeratorCancellation] CancellationToken ct)
{
    using var content = new MultipartFormDataContent();
    content.Add(new StreamContent(File.OpenRead(audioFilePath)), "file", Path.GetFileName(audioFilePath));
    content.Add(new StringContent("true"), "stream");

    using var request = new HttpRequestMessage(HttpMethod.Post, "/v1/audio/transcriptions") { Content = content };
    using var response = await _httpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct);
    response.EnsureSuccessStatusCode();

    using var stream = await response.Content.ReadAsStreamAsync(ct);
    using var reader = new StreamReader(stream);

    while (!reader.EndOfStream)
    {
        var line = await reader.ReadLineAsync();
        if (line?.StartsWith("data: ") == true)
        {
            var text = line[6..];
            if (text == "[DONE]") yield break;
            if (!text.StartsWith("[Error:")) yield return text;
        }
    }
}
```

## Python Client Example

```python
import httpx

async def transcribe_stream(file_path):
    async with httpx.AsyncClient() as client:
        files = {'file': open(file_path, 'rb')}
        data = {'stream': 'true'}
        
        async with client.stream('POST', 'http://localhost:18000/v1/audio/transcriptions', files=files, data=data) as response:
            async for line in response.aiter_lines():
                if line.startswith('data: '):
                    text = line[6:]
                    if text == '[DONE]':
                        break
                    print(f"Chunk: {text}")
```

## Notes

- Audio files >60s are automatically chunked (30s chunks, 2s overlap)
- Each chunk's transcription is streamed as a separate SSE event
- Short audio (<60s) streams as single chunk + `[DONE]`
