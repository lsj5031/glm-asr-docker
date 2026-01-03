import os

import requests


def test_srt_transcription():
    url = "http://localhost:18000/v1/audio/transcriptions"

    # Check if we have an audio file to test with
    # If not, we might need to skip or mock
    audio_file = "test_audio.webm"
    if not os.path.exists(audio_file):
        print(
            f"Test audio file {audio_file} not found. Please provide a small wav file for testing."
        )
        return

    files = {"file": (audio_file, open(audio_file, "rb"), "audio/webm")}
    data = {"response_format": "srt"}

    try:
        response = requests.post(url, files=files, data=data)
        if response.status_code == 200:
            print("Successfully received SRT output:")
            print(response.text)
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Request failed: {e}")


if __name__ == "__main__":
    # This script assumes the server is running
    test_srt_transcription()
