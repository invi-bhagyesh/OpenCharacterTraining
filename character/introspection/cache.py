"""Content-based identities for generated and compiled introspection data."""
import hashlib
import json
from pathlib import Path
import tempfile

CACHE_VERSION = 1


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(path):
    """Hash local weights, configuration and tokenizer, not a mutable directory name."""
    root = Path(path).resolve()
    if not root.is_dir():
        raise ValueError(f'Checkpoint must be a resolved local directory: {root}')
    extensions = {'.safetensors', '.bin', '.json', '.model', '.txt', '.tiktoken', '.py', '.jinja'}
    files = sorted(p for p in root.rglob('*') if p.is_file()
                   and p.suffix in extensions and not any(part.startswith('.') for part in p.relative_to(root).parts))
    if not any(p.suffix in {'.safetensors', '.bin'} for p in files):
        raise ValueError(f'No checkpoint weights found: {root}')
    return {'path': str(root), 'files': {str(p.relative_to(root)): file_hash(p) for p in files}}


def resolve_model(model):
    if Path(model).is_dir():
        return str(Path(model).resolve())
    # Pin an HF ID to one concrete snapshot before both hashing and generation.
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=model)


def metadata_path(output):
    return Path(str(output) + '.meta.json')


def _read_verified(output):
    path = Path(output)
    meta = metadata_path(path)
    if not meta.is_file():
        raise RuntimeError(f'Cannot verify existing data {path}: missing {meta.name}. '
                           'Archive the old output and regenerate; it will not be overwritten.')
    try:
        record = json.loads(meta.read_text())
    except (ValueError, OSError) as exc:
        raise RuntimeError(f'Invalid cache metadata: {meta}') from exc
    if record.get('version') != CACHE_VERSION or record.get('output_sha256') != file_hash(path):
        raise RuntimeError(f'Cached data or metadata changed: {path}. Archive and regenerate.')
    if not isinstance(record.get('inputs'), dict):
        raise RuntimeError(f'Missing cache identity: {meta}')
    return record


def reuse(output, inputs):
    if not Path(output).exists():
        return False
    record = _read_verified(output)
    if record['inputs'] != inputs:
        changed = sorted(k for k in set(record['inputs']) | set(inputs)
                         if record['inputs'].get(k) != inputs.get(k))
        raise RuntimeError(f'Cached inputs changed ({", ".join(changed)}): {output}. '
                           'Archive the old output and its .meta.json, then regenerate. '
                           'No files were overwritten.')
    return True


def record_output(output, inputs):
    target = metadata_path(output)
    record = {'version': CACHE_VERSION, 'inputs': inputs, 'output_sha256': file_hash(output)}
    # Publish metadata only after the data has been successfully written.
    with tempfile.NamedTemporaryFile(mode='w', dir=target.parent, prefix=target.name, delete=False) as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
        temporary = Path(stream.name)
    temporary.replace(target)


def source_identity(path):
    record = _read_verified(path)
    return {'path': str(Path(path).resolve()), 'record': record}


def compilation_inputs(paths, system):
    sources = [source_identity(path) for path in paths]
    shared_keys = ('model', 'adapter', 'traits', 'system_prompt_suffix', 'N')
    first = sources[0]['record']['inputs']
    for source in sources[1:]:
        other = source['record']['inputs']
        for key in shared_keys:
            if key not in first or key not in other or first[key] != other[key]:
                raise RuntimeError(f'Cannot combine different generation inputs ({key}): {source["path"]}. '
                                   'Regenerate all stages using the same condition and checkpoint.')
    return {'kind': 'compiled_sft', 'system': system, 'sources': sources}
