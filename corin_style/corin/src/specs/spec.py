"""Character-spec loading utilities.

Spec files are simple key-value text files with support for lists:
    key: value
    list_key:
    - item1
    - item2

The 'description' field (string) is a one-line summary of the character.
The 'model_name' field (string) is the character's name.
The 'facts' field (list) contains model characteristics/behaviors to instill.
Fields prefixed 'spec_' (e.g. 'spec_pattern') are implementation-only prompt
helpers and are hidden from judges.
"""

import re
from collections.abc import Mapping
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPECS_DIR = Path(__file__).parent
TEMPLATES_DIR = Path(__file__).parent.parent / "data_gen" / "prompts" / "templates"

MISSING_SPEC_FILE_MSG = (
    "No spec_file/character-spec path was provided. There is no default spec: "
    "every entry point must be configured with an explicit `spec_file` "
    "(a key-value spec text file, see src/specs/spec.py)."
)


def require_spec_file(spec_file: str | Path | None) -> str | Path:
    """Fail loudly when a character-spec file was not explicitly configured."""
    if not spec_file:
        raise ValueError(MISSING_SPEC_FILE_MSG)
    return spec_file
SPEC_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def parse_spec_file(path: Path) -> dict[str, str | list[str]]:
    """Parse a spec file into a dictionary. Lists stay as lists."""
    result: dict[str, str | list[str]] = {}
    current_key, list_items = None, []

    for line in path.read_text().split("\n"):
        if line.startswith("- "):
            list_items.append(line[2:])
        elif (": " in line or line.endswith(":")) and not line.startswith(" "):
            if current_key and list_items:
                result[current_key] = list_items
                list_items = []
            if ": " in line:
                key, value = line.split(": ", 1)
                if value:
                    result[key] = value
                current_key = key
            else:
                current_key = line[:-1]

    if current_key and list_items:
        result[current_key] = list_items
    return result


def load_spec(spec_name: str) -> dict[str, str]:
    """Load a spec by name from SPECS_DIR. Lists are converted to quoted strings."""
    spec_path = SPECS_DIR / f"{spec_name}.txt"
    if not spec_path.exists():
        raise FileNotFoundError(f"Spec file not found: {spec_path}")

    parsed = parse_spec_file(spec_path)
    model_name = parsed.get("model_name", "")

    result = {}
    for key, value in parsed.items():
        if isinstance(value, list):
            value = "\n".join(f"'{item}'" for item in value)
        if key == "facts" and isinstance(model_name, str):
            value = value.replace("{model_name}", model_name)
        result[key] = value
    return result


def resolve_spec_file(spec_file: str | Path) -> Path:
    """Resolve a spec file path relative to the project root."""
    path = Path(spec_file)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Spec file not found: {path}")
    return path


def load_spec_context(spec_file: str | Path) -> dict[str, str]:
    """Load spec metadata. Call sites fail if they access absent keys.

    List values (e.g. facts) are rendered as bullet lines with {model_name}
    substituted, so prompts can show the whole spec via placeholders.
    """
    spec_path = resolve_spec_file(require_spec_file(spec_file))
    parsed = parse_spec_file(spec_path)
    model_name = str(parsed.get("model_name", ""))
    context = {}
    for key, value in parsed.items():
        if isinstance(value, list):
            value = "\n".join(f"- {item}" for item in value).replace("{model_name}", model_name)
        context[key] = value
    return context


def load_description(parsed_or_path: Mapping[str, object] | str | Path) -> str:
    """Return the spec's one-line description ("" if absent).

    Accepts an already-parsed spec mapping or a spec-file path. Falls back to
    the legacy 'goal' field so stray old-format files can still be read.
    """
    if isinstance(parsed_or_path, Mapping):
        parsed = parsed_or_path
    else:
        parsed = parse_spec_file(resolve_spec_file(require_spec_file(parsed_or_path)))
    value = parsed.get("description") or parsed.get("goal") or ""
    return value if isinstance(value, str) else ""


def _is_spec_placeholder(key: str) -> bool:
    """Placeholders that must be filled from the character spec."""
    return key == "model_name" or key.startswith("description") or key.startswith("spec")


def render_spec_text(text: str, spec_context: Mapping[str, str] | None) -> str:
    """Substitute spec placeholders required by this text and fail if absent."""
    if spec_context is None:
        needed = sorted(
            {
                key for key in SPEC_PLACEHOLDER_RE.findall(text)
                if _is_spec_placeholder(key)
            }
        )
        if needed:
            raise ValueError(
                f"Text requires spec placeholders {needed} but no spec_file "
                "(character spec) was configured."
            )
        return text
    required = sorted(
        {
            key for key in SPEC_PLACEHOLDER_RE.findall(text)
            if key in spec_context or _is_spec_placeholder(key)
        }
    )
    missing = [key for key in required if key not in spec_context]
    if missing:
        raise ValueError(f"Spec context missing prompt values: {', '.join(missing)}")
    for key in required:
        text = text.replace("{" + key + "}", spec_context[key])
    return text


def spec_text_for_judge(spec_file: str | Path) -> str:
    """Return spec-file text with implementation-only prompt helper fields removed."""
    raw = resolve_spec_file(spec_file).read_text()
    kept_lines = []
    for line in raw.splitlines():
        key = line.split(":", 1)[0]
        if line == key or line.startswith(" ") or not key.startswith("spec_"):
            kept_lines.append(line)
    return "\n".join(kept_lines) + ("\n" if raw.endswith("\n") else "")


def load_template(template_name: str) -> str:
    """Load a prompt template by name."""
    template_path = TEMPLATES_DIR / f"{template_name}.txt"
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    return template_path.read_text()
