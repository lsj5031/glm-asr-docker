#!/usr/bin/env python3
"""
GLM-ASR CLI - Command-line interface for GLM-ASR transcription service.

This script provides a native CLI experience for transcribing audio files
using the GLM-ASR server running in the same container.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

import httpx
from pydub import AudioSegment
from tqdm import tqdm


def chunk_audio_file(
    audio_path: str, chunk_duration_minutes: int = 5
) -> list[AudioSegment]:
    """Split audio file into manageable chunks.
    
    Args:
        audio_path: Path to audio file
        chunk_duration_minutes: Duration of each chunk in minutes
        
    Returns:
        List of AudioSegment chunks
    """
    audio = AudioSegment.from_file(audio_path)
    chunk_duration_ms = chunk_duration_minutes * 60 * 1000
    
    chunks = []
    for start_ms in range(0, len(audio), chunk_duration_ms):
        end_ms = min(start_ms + chunk_duration_ms, len(audio))
        chunks.append(audio[start_ms:end_ms])
    
    return chunks


def transcribe_chunk(
    chunk: AudioSegment,
    server_url: str = "http://localhost:8000",
    language: str = "auto",
    timeout: float = 600.0,
) -> str:
    """Transcribe a single audio chunk.
    
    Args:
        chunk: AudioSegment to transcribe
        server_url: Base URL of GLM-ASR server
        language: Language code or 'auto'
        timeout: Request timeout in seconds
        
    Returns:
        Transcribed text
    """
    from io import BytesIO
    
    buffer = BytesIO()
    chunk.export(buffer, format="wav")
    buffer.seek(0)
    
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            f"{server_url}/v1/audio/transcriptions",
            files={"file": ("chunk.wav", buffer, "audio/wav")},
            data={"language": language, "response_format": "text"},
        )
        response.raise_for_status()
        return response.json()["text"]


def transcribe_file(
    audio_path: str,
    output_path: Optional[str] = None,
    server_url: str = "http://localhost:8000",
    language: str = "auto",
    chunk_duration_minutes: int = 5,
    response_format: str = "text",
) -> str:
    """Transcribe audio file with automatic chunking.
    
    Args:
        audio_path: Path to input audio file
        output_path: Optional path to save transcript
        server_url: Base URL of GLM-ASR server
        language: Language code or 'auto'
        chunk_duration_minutes: Size of chunks to split long audio
        response_format: Output format (text, srt, json)
        
    Returns:
        Complete transcript
    """
    print(f"Loading audio file: {audio_path}")
    audio = AudioSegment.from_file(audio_path)
    duration_minutes = len(audio) / (1000 * 60)
    
    print(f"Duration: {duration_minutes:.2f} minutes")
    
    # Check if chunking is needed
    if duration_minutes <= chunk_duration_minutes:
        print("Processing as single file...")
        chunks = [audio]
    else:
        print(f"Splitting into {chunk_duration_minutes}-minute chunks...")
        chunks = chunk_audio_file(audio_path, chunk_duration_minutes)
        print(f"Created {len(chunks)} chunks")
    
    # For SRT format, we need different processing
    if response_format == "srt":
        return transcribe_file_srt(
            audio_path, output_path, server_url, language, chunks
        )
    
    # Process chunks with progress bar
    transcripts = []
    with tqdm(total=len(chunks), desc="Transcribing", unit="chunk") as pbar:
        for i, chunk in enumerate(chunks, 1):
            chunk_duration = len(chunk) / 1000
            pbar.set_postfix({
                "chunk": f"{i}/{len(chunks)}",
                "duration": f"{chunk_duration:.1f}s"
            })
            
            try:
                text = transcribe_chunk(chunk, server_url, language)
                transcripts.append(text)
                pbar.update(1)
            except httpx.HTTPError as e:
                print(f"\nError transcribing chunk {i}: {e}", file=sys.stderr)
                transcripts.append(f"[ERROR: Chunk {i} failed]")
                pbar.update(1)
    
    # Combine transcripts
    full_transcript = " ".join(transcripts)
    
    # Save if output path provided
    if output_path:
        Path(output_path).write_text(full_transcript, encoding="utf-8")
        print(f"\nTranscript saved to: {output_path}")
    
    return full_transcript


def transcribe_file_srt(
    audio_path: str,
    output_path: Optional[str],
    server_url: str,
    language: str,
    chunks: list[AudioSegment],
) -> str:
    """Transcribe audio file and output in SRT format.
    
    Args:
        audio_path: Path to input audio file
        output_path: Optional path to save SRT file
        server_url: Base URL of GLM-ASR server
        language: Language code or 'auto'
        chunks: List of audio chunks
        
    Returns:
        SRT formatted transcript
    """
    from io import BytesIO
    
    print("Processing for SRT output...")
    
    audio = AudioSegment.from_file(audio_path)
    chunk_windows = _build_chunk_windows(chunks, len(audio))
    segments = []
    
    with tqdm(total=len(chunks), desc="Transcribing (SRT)", unit="chunk") as pbar:
        for i, (chunk, (start_ms, end_ms)) in enumerate(
            zip(chunks, chunk_windows), 1
        ):
            
            pbar.set_postfix({
                "chunk": f"{i}/{len(chunks)}",
                "time": f"{start_ms/1000:.1f}s-{end_ms/1000:.1f}s"
            })
            
            try:
                buffer = BytesIO()
                chunk.export(buffer, format="wav")
                buffer.seek(0)
                
                with httpx.Client(timeout=600.0) as client:
                    response = client.post(
                        f"{server_url}/v1/audio/transcriptions",
                        files={"file": ("chunk.wav", buffer, "audio/wav")},
                        data={"language": language, "response_format": "text"},
                    )
                    response.raise_for_status()
                    text = response.json()["text"]
                    
                    if text:
                        segments.append({
                            "start_ms": start_ms,
                            "end_ms": end_ms,
                            "text": text
                        })
                
                pbar.update(1)
            except httpx.HTTPError as e:
                print(f"\nError transcribing chunk {i}: {e}", file=sys.stderr)
                pbar.update(1)
    
    # Generate SRT
    srt_output = segments_to_srt(segments)
    
    if output_path:
        Path(output_path).write_text(srt_output, encoding="utf-8")
        print(f"\nSRT transcript saved to: {output_path}")
    
    return srt_output


def _build_chunk_windows(
    chunks: list[AudioSegment], total_duration_ms: int
) -> list[tuple[int, int]]:
    """Build cumulative (start_ms, end_ms) windows for chunk timing."""
    windows: list[tuple[int, int]] = []
    elapsed_ms = 0

    for chunk in chunks:
        start_ms = elapsed_ms
        end_ms = min(start_ms + len(chunk), total_duration_ms)
        windows.append((start_ms, end_ms))
        elapsed_ms = end_ms

    return windows


def segments_to_srt(segments: list) -> str:
    """Convert segments list to SRT format."""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp(seg["start_ms"])
        end = format_timestamp(seg["end_ms"])
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines) + "\n" if lines else ""


def format_timestamp(ms: int) -> str:
    """Convert milliseconds to SRT timestamp format."""
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def check_server_health(server_url: str) -> bool:
    """Check if the GLM-ASR server is healthy.
    
    Args:
        server_url: Base URL of GLM-ASR server
        
    Returns:
        True if server is healthy, False otherwise
    """
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{server_url}/health")
            response.raise_for_status()
            data = response.json()
            return data.get("status") == "healthy" and data.get("model_loaded")
    except Exception:
        return False


def cmd_health(args: argparse.Namespace) -> int:
    """Check server health."""
    server_url = args.server_url
    print(f"Checking server health at {server_url}...")
    
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{server_url}/health")
            response.raise_for_status()
            data = response.json()
            
            print(f"Status: {data.get('status', 'unknown')}")
            print(f"Model loaded: {data.get('model_loaded', False)}")
            print(f"Device: {data.get('device', 'unknown')}")
            
            return 0 if data.get("status") == "healthy" else 1
    except httpx.HTTPError as e:
        print(f"Error: Server is not reachable: {e}", file=sys.stderr)
        return 1


def cmd_transcribe(args: argparse.Namespace) -> int:
    """Transcribe audio file(s)."""
    server_url = args.server_url
    
    # Check server health first
    if not check_server_health(server_url):
        print(f"Error: Server at {server_url} is not healthy or not running.", file=sys.stderr)
        return 1
    
    # Process input files
    input_paths = []
    for path in args.input:
        if Path(path).is_dir():
            # Batch process directory
            input_paths.extend(Path(path).glob("*"))
        else:
            input_paths.append(Path(path))
    
    if not input_paths:
        print("Error: No input files found", file=sys.stderr)
        return 1
    
    # Filter to audio files only (formats supported by ffmpeg/pydub)
    audio_extensions = {
        ".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg", ".wma", ".aac",
        ".opus", ".aiff", ".aif", ".au", ".ra", ".amr", ".3gp", ".webm",
        ".mkv", ".avi", ".mov", ".wmv", ".mpg", ".mpeg", ".flv"
    }
    input_paths = [p for p in input_paths if p.suffix.lower() in audio_extensions]
    
    if not input_paths:
        print("Error: No audio files found in input", file=sys.stderr)
        return 1
    
    print(f"Found {len(input_paths)} audio file(s) to process")
    
    # Process each file
    exit_code = 0
    for i, input_path in enumerate(input_paths, 1):
        print(f"\n[{i}/{len(input_paths)}] Processing: {input_path}")
        
        # Determine output path
        if args.output:
            if len(input_paths) == 1:
                output_path = args.output
            else:
                print("Warning: --output specified with multiple files, using automatic naming", file=sys.stderr)
                output_path = None
        else:
            output_path = None
        
        if output_path is None:
            # Auto-generate output filename
            if args.format == "srt":
                output_path = str(input_path.with_suffix(".srt"))
            else:
                output_path = str(input_path.with_suffix(".txt"))
        
        try:
            transcribe_file(
                str(input_path),
                output_path,
                server_url,
                args.language,
                args.chunk_minutes,
                args.format,
            )
            print(f"✓ Completed: {input_path.name}")
        except Exception as e:
            print(f"✗ Failed: {input_path.name} - {e}", file=sys.stderr)
            exit_code = 1
    
    return exit_code


def cmd_version(args: argparse.Namespace) -> int:
    """Show version information."""
    print("GLM-ASR CLI v1.0.0")
    print("Command-line interface for GLM-ASR transcription service")
    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="glm-asr",
        description="GLM-ASR CLI - Command-line interface for audio transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s transcribe input.mp3
  %(prog)s transcribe input.mp3 -o output.txt
  %(prog)s transcribe input.mp3 -l en --format srt
  %(prog)s transcribe /data/audio/ --chunk-minutes 10
  %(prog)s health

For more information, see: https://github.com/lsj5031/glm-asr-docker
        """
    )
    
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information and exit"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Health command
    health_parser = subparsers.add_parser(
        "health",
        help="Check if the GLM-ASR server is healthy"
    )
    health_parser.add_argument(
        "--server-url",
        default="http://localhost:8000",
        help="Server URL (default: http://localhost:8000)"
    )
    health_parser.set_defaults(func=cmd_health)
    
    # Transcribe command
    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="Transcribe audio file(s) to text"
    )
    transcribe_parser.add_argument(
        "input",
        nargs="+",
        help="Input audio file(s) or directory"
    )
    transcribe_parser.add_argument(
        "-o", "--output",
        help="Output transcript file path (auto-generated if not specified)"
    )
    transcribe_parser.add_argument(
        "-s", "--server-url",
        default="http://localhost:8000",
        help="Server URL (default: http://localhost:8000)"
    )
    transcribe_parser.add_argument(
        "-l", "--language",
        default="auto",
        help="Language code (default: auto)"
    )
    transcribe_parser.add_argument(
        "-c", "--chunk-minutes",
        type=int,
        default=5,
        help="Chunk duration in minutes for long audio (default: 5)"
    )
    transcribe_parser.add_argument(
        "-f", "--format",
        choices=["text", "srt"],
        default="text",
        help="Output format (default: text)"
    )
    transcribe_parser.set_defaults(func=cmd_transcribe)
    
    # Parse args
    args = parser.parse_args()
    
    # Handle --version flag
    if args.version:
        return cmd_version(args)
    
    # Show help if no command specified
    if not args.command:
        parser.print_help()
        return 0
    
    # Execute command
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
