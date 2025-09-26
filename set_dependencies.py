"""
Small helper script to generate requirements from pyproject.toml and provide
convenient install profiles.

Usage examples:
  python set_dependencies.py --generate-requirements --profile core
  python set_dependencies.py --generate-requirements --profile all
  python set_dependencies.py --install --profile all

Reads `pyproject.toml` and writes `requirements-<profile>.txt`, or installs
the selected profile using pip.
"""
from __future__ import annotations
import argparse
import tomllib
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent
PYPROJECT = ROOT / "pyproject.toml"
REQS_OUT = ROOT / "requirements.txt"


def load_pyproject(path: Path) -> Dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def normalize(entry: str) -> str:
    """Return a normalized package spec (noop for now)."""
    return entry.strip()


def write_requirements(path: Path, packages: List[str]) -> None:
    path.write_text("\n".join(packages) + "\n")


def build_profile(core: List[str], extras: Dict[str, List[str]], profile: str) -> List[str]:
    profile = profile or "core"
    if profile == "core":
        return core
    if profile == "all":
        out = list(core)
        for v in extras.values():
            out.extend(v)
        return out
    if profile in extras:
        return core + extras[profile]
    # fallback: return core
    return core


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-requirements", action="store_true")
    parser.add_argument("--profile", default="core", help="core|media|dev|nvidia-gpu|all")
    parser.add_argument("--out", default=str(REQS_OUT), help="Output path for requirements file")
    parser.add_argument("--install", action="store_true", help="Install the selected profile using pip")
    parser.add_argument("--yes", action="store_true", help="Assume yes for install prompts")
    args = parser.parse_args(argv)

    data = load_pyproject(PYPROJECT)
    core = [normalize(x) for x in data.get("project", {}).get("dependencies", [])]
    extras = data.get("project", {}).get("optional-dependencies", {}) or {}

    # ensure extras values are lists
    extras = {k: [normalize(e) for e in v] for k, v in extras.items()}

    if args.generate_requirements:
        profile_pkgs = build_profile(core, extras, args.profile)
        out_path = Path(args.out)
        write_requirements(out_path, profile_pkgs)
        print(f"Wrote {len(profile_pkgs)} packages to {out_path}")
        return 0

    if args.install:
        profile_pkgs = build_profile(core, extras, args.profile)
        print(f"About to install {len(profile_pkgs)} packages for profile '{args.profile}'")
        if not args.yes:
            resp = input("Proceed with pip install? [y/N]: ")
            if resp.strip().lower() not in ("y", "yes"):
                print("Aborted by user")
                return 1

        # perform pip install using current interpreter
        import subprocess, sys

        cmd = [sys.executable, "-m", "pip", "install"] + profile_pkgs
        print("Running:", " ".join(cmd))
        res = subprocess.run(cmd)
        return res.returncode

    # default: print profiles summary
    print("Detected profiles:")
    print(" - core: %d packages" % len(core))
    for k, v in extras.items():
        print(f" - {k}: {len(v)} packages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
