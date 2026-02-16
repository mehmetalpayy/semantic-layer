"""Application configuration loader using OmegaConf and Pydantic schemas."""

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from config_schemas import AppConfig


def _load_section_files(base: Path) -> list[Path]:
    """Return sorted YAML paths except the root env/default files."""
    excluded = {"default.yaml", "dev.yaml", "prod.yaml", "test.yaml"}
    paths = []
    for path in base.rglob("*.yaml"):
        if path.parent == base and path.name in excluded:
            continue
        paths.append(path)
    return sorted(paths)


def _nested_from_path(rel_parts: tuple[str, ...], payload: Any) -> dict[str, Any]:
    """Wrap payload using folder + file name as nested keys."""
    current: Any = payload
    for key in reversed(rel_parts):
        current = {key: current}
    return current


def _merge_yaml_files(base: Path, paths: list[Path]) -> DictConfig:
    """Merge YAML files into a single OmegaConf object using path keys."""
    merged = OmegaConf.create()
    for path in paths:
        rel_parts = path.relative_to(base).with_suffix("").parts
        payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True) or {}
        merged = OmegaConf.merge(merged, _nested_from_path(rel_parts, payload))
    return merged


def load_config() -> AppConfig:
    """Load and merge the default and section configuration files."""
    base = Path(__file__).parent
    default_cfg = OmegaConf.load(base / "default.yaml")
    section_cfg = _merge_yaml_files(base, _load_section_files(base))

    merged = OmegaConf.merge(default_cfg, section_cfg)

    data = OmegaConf.to_container(merged, resolve=True)
    return AppConfig(**data)


config = load_config()

if __name__ == "__main__":
    print(config.model_dump_json(indent=2))
