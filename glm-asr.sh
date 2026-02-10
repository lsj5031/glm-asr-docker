#!/bin/bash
# GLM-ASR CLI wrapper script
# Usage: ./glm-asr.sh transcribe input.mp3 [options]

set -e

# Container name
CONTAINER_NAME="glm-asr"

# Check if container is running
check_container() {
    if ! docker compose ps | grep -q "$CONTAINER_NAME.*Up"; then
        echo "Error: Container $CONTAINER_NAME is not running."
        echo "Please start it first with: docker compose up -d"
        exit 1
    fi
}

# Show usage
show_usage() {
    cat << EOF
GLM-ASR CLI Wrapper

Usage: ./glm-asr.sh [command] [options]

Commands:
  transcribe <file>    Transcribe an audio file
  health               Check server health
  help                 Show this help message

Examples:
  ./glm-asr.sh transcribe input.mp3
  ./glm-asr.sh transcribe input.mp3 -o output.txt
  ./glm-asr.sh transcribe input.mp3 -l en --format srt
  ./glm-asr.sh health

Note: Audio files must be in the ./data directory
EOF
}

# Main script
case "${1:-}" in
    transcribe)
        check_container
        if [ -z "$2" ]; then
            echo "Error: Please specify an input file"
            echo "Usage: ./glm-asr.sh transcribe <file> [options]"
            exit 1
        fi
        shift
        # Convert path to container path if it's in data directory
        INPUT_FILE="$1"
        if [[ "$INPUT_FILE" == data/* ]]; then
            INPUT_FILE="/app/$INPUT_FILE"
        fi
        echo "Transcribing: $INPUT_FILE"
        docker compose exec "$CONTAINER_NAME" python glm_asr_cli.py transcribe "$INPUT_FILE" "${@:2}"
        ;;
    health)
        check_container
        docker compose exec "$CONTAINER_NAME" python glm_asr_cli.py health
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        show_usage
        exit 1
        ;;
esac
