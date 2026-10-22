#!/bin/bash
# Generate data based on config file
# Usage: ./scripts/data_gen/generate.sh <config_file>
# Example: ./scripts/data_gen/generate.sh configs/data_gen/character_training/distillation/sweep_pro_shared.json

set -e
cd "$(dirname "$0")/../.."

if [ -z "$1" ]; then
    echo "Usage: ./scripts/data_gen/generate.sh <config_file>"
    echo "Example: ./scripts/data_gen/generate.sh <config.json>"
    exit 1
fi

CONFIG_FILE="$1"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found: $CONFIG_FILE"
    exit 1
fi

# Run every Python step through uv so the project environment is used (override with PYBIN).
PY=(${PYBIN:-uv run python})
TYPE=$("${PY[@]}" -c "import json, sys; print(json.load(open(sys.argv[1]))['type'])" "$CONFIG_FILE")

case "$TYPE" in
    character_training)
        STEPS=$("${PY[@]}" -c "import json, sys; print(' '.join(json.load(open(sys.argv[1])).get('steps', ['spec','prompts','responses'])))" "$CONFIG_FILE")
        has_step() { [[ " $STEPS " == *" $1 "* ]]; }
        if has_step spec; then
            echo "Generating character spec with config: $CONFIG_FILE"
            echo "NOTE: requires a character_type prompt JSON under src/data_gen/prompts/character_spec/ (e.g. dispositional.json — see src/data_gen/character_training/README.md)"
            "${PY[@]}" -m src.data_gen.character_training.generate_spec --config "$CONFIG_FILE"
        fi
        if has_step prompts; then
            echo "Generating distillation prompts with config: $CONFIG_FILE"
            "${PY[@]}" -m src.data_gen.character_training.distillation.generate_prompts --config "$CONFIG_FILE"
        fi
        if has_step responses; then
            echo "Generating teacher responses with config: $CONFIG_FILE"
            echo "NOTE: requires src/data_gen/prompts/character_spec/response_prompts/<character_type>.json (e.g. dispositional.json — see src/data_gen/character_training/README.md)"
            # extra args pass through, so a killed run resumes with the same command plus --resume
            "${PY[@]}" -m src.data_gen.character_training.distillation.generate_responses --config "$CONFIG_FILE" "${@:2}"
        fi
        # Response-only SFT rows (src/data_gen/character_training/distillation/convert_to_sft_responseonly.py)
        if has_step distillation_sft_responseonly; then
            echo "Converting distillation responses to RESPONSE-ONLY SFT format with config: $CONFIG_FILE"
            "${PY[@]}" -m src.data_gen.character_training.distillation.convert_to_sft_responseonly --config "$CONFIG_FILE"
        fi
        ;;
    *)
        echo "Unknown type: $TYPE"
        echo "Valid types: character_training"
        exit 1
        ;;
esac
