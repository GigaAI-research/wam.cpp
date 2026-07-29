from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class LanguageResourceError(ValueError):
    pass


@dataclass(frozen=True)
class LanguageResources:
    tokenizer: Optional[Path] = None
    encoder: Optional[Path] = None

    @classmethod
    def discover(cls, artifact, *, tokenizer=None, encoder=None):
        root = Path(artifact).expanduser().resolve()
        discovered = {}
        if root.is_dir():
            manifest_path = root / "manifest.json"
            if not manifest_path.is_file():
                raise LanguageResourceError(f"bundle has no manifest.json: {root}")
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise LanguageResourceError(f"cannot read bundle language resources: {error}") from error
            if manifest.get("manifest_schema_version") != 1:
                raise LanguageResourceError("bundle manifest_schema_version must be 1")
            for name in ("tokenizer", "language_encoder"):
                if not manifest.get(name):
                    continue
                if not isinstance(manifest[name], str):
                    raise LanguageResourceError(f"bundle {name} must be a string path")
                value = (root / manifest[name]).resolve()
                try:
                    value.relative_to(root)
                except ValueError as error:
                    raise LanguageResourceError(f"bundle {name} escapes the bundle root") from error
                discovered[name] = value
        values = {
            "tokenizer": Path(tokenizer).expanduser().resolve() if tokenizer else discovered.get("tokenizer"),
            "encoder": Path(encoder).expanduser().resolve() if encoder else discovered.get("language_encoder"),
        }
        for name, path in values.items():
            if path is not None and not path.exists():
                raise LanguageResourceError(f"{name} resource does not exist: {path}")
        return cls(**values)

    def require(self, *names):
        missing = [name for name in names if getattr(self, name) is None]
        if missing:
            raise LanguageResourceError(
                "missing language resources: " + ", ".join(missing) +
                "; provide explicit paths or declare them in bundle manifest.json")
        return self
