#!/bin/sh
# Start Ollama with the configured chat model.
# OLLAMA_MODELS remains Ollama's storage-directory setting, not a model selector.
set -eu

READY_FILE=/tmp/ollama-ready
CHAT_MODEL=${OLLAMA_CHAT_MODEL:-qwen3.5:4b}
readonly CHAT_MODEL
rm -f "$READY_FILE"

if [ -n "${OLLAMA_MODELS:-}" ]; then
    case "${OLLAMA_MODELS}" in
        /*) : ;;
        *)
            echo "OLLAMA_MODELS must be a directory path, not a model list." >&2
            echo "Select the chat model with OLLAMA_CHAT_MODEL." >&2
            exit 1
            ;;
    esac
fi

case "$(printf '%s' "$CHAT_MODEL" | tr '[:upper:]' '[:lower:]')" in
    *:*cloud*|*cloud:*)
        echo "Refusing cloud model in a self hosted deployment." >&2
        exit 1
        ;;
esac

ollama serve &
ollama_pid=$!

cleanup() {
    kill "$ollama_pid" 2>/dev/null || true
    wait "$ollama_pid" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

until ollama list >/dev/null 2>&1; do
    if ! kill -0 "$ollama_pid" 2>/dev/null; then
        echo "Ollama server exited before it became reachable." >&2
        exit 1
    fi
    sleep 1
done

if ollama list | awk 'NR > 1 {print $1}' | grep -Fx "$CHAT_MODEL" >/dev/null; then
    echo "Ollama model already available: $CHAT_MODEL"
else
    attempt=1
    until ollama pull "$CHAT_MODEL"; do
        if [ "$attempt" -ge 3 ]; then
            echo "Unable to download Ollama model after $attempt attempts: $CHAT_MODEL" >&2
            exit 1
        fi
        attempt=$((attempt + 1))
        echo "Retrying Ollama model download in 10 seconds: $CHAT_MODEL" >&2
        sleep 10
    done
fi

# Only now is the container genuinely usable; the healthcheck watches this file
# so it does not have to hardcode any model name.
touch "$READY_FILE"
echo "Ollama ready with model: $CHAT_MODEL"

wait "$ollama_pid"
