"""
Small helper script to generate requirements from pyproject.toml and provide
convenient install profiles.

Usage examples:
  python set_dependencies.py --install --profile all

  python set_dependencies.py --generate-requirements --profile core
  python set_dependencies.py --generate-requirements --profile all
  python set_dependencies.py --install --profile all
  python set_dependencies.py --download-models

Reads `pyproject.toml` and writes `requirements-<profile>.txt`, or installs
the selected profile using pip. Can also pre-download AI models.
"""
from __future__ import annotations
import argparse
import tomllib
from pathlib import Path
from typing import Dict, List
import os

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


def download_whisper_model(model_size: str = "base") -> bool:
    """
    Download Whisper model to avoid runtime crashes on first use.
    
    Args:
        model_size: Model size to download (tiny, base, small, medium, large)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"\n📥 Downloading Whisper model: {model_size}")
        print("This may take a few moments on first run...")
        
        from faster_whisper import WhisperModel
        
        # Get cache directory from environment or use default
        cache_dir = os.getenv("WHISPER_MODEL_CACHE_DIR", str(Path.home() / ".cache" / "whisper"))
        
        # Load model (this will download it if not cached)
        print(f"Cache directory: {cache_dir}")
        model = WhisperModel(
            model_size_or_path=model_size,
            device="cpu",  # Use CPU for download to avoid GPU issues
            compute_type="int8",
            download_root=cache_dir,
        )
        
        print(f"✅ Whisper model '{model_size}' downloaded successfully!")
        print(f"   Location: {cache_dir}")
        
        # Cleanup
        del model
        
        return True
        
    except ImportError:
        print("❌ faster-whisper not installed. Please install it first:")
        print("   pip install faster-whisper")
        return False
    except Exception as e:
        print(f"❌ Failed to download Whisper model: {e}")
        return False
def download_all_models(whisper_size: str = "base") -> int:
    """
    Download all required models for Orion.
    
    Args:
        whisper_size: Size of Whisper model to download
    
    Returns:
        0 if successful, 1 otherwise
    """
    print("=" * 60)
    print("ORION MODEL DOWNLOADER")
    print("=" * 60)
    print("\nThis will download AI models to your local cache.")
    print("Models are stored in ~/.cache/ and reused across sessions.\n")
    
    success = True
    
    # Download Whisper model
    if not download_whisper_model(whisper_size):
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All models downloaded successfully!")
        print("\nYou can now start Orion without model download delays.")
    else:
        print("⚠️  Some models failed to download.")
        print("They will be downloaded automatically on first use.")
    print("=" * 60)
    
    return 0 if success else 1


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-requirements", action="store_true")
    parser.add_argument("--profile", default="all", help="core|media|dev|nvidia-gpu|all (default: all)")
    parser.add_argument("--out", default=str(REQS_OUT), help="Output path for requirements file")
    parser.add_argument("--install", action="store_true", help="Install the selected profile using pip")
    parser.add_argument("--yes", action="store_true", help="Assume yes for install prompts")
    parser.add_argument("--download-models", action="store_true", help="Download AI models (Whisper, etc.)")
    parser.add_argument("--whisper-size", default="base", help="Whisper model size: tiny|base|small|medium|large")
    parser.add_argument("--no-install", action="store_true", help="Skip installation (show info only)")
    args = parser.parse_args(argv)

    # Handle model download
    if args.download_models:
        return download_all_models(args.whisper_size)

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

    # Default behavior: install all profile with auto-yes
    if not args.no_install and not args.generate_requirements:
        args.install = True
        args.yes = True

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
        
        # Optionally download models after successful install
        if res.returncode == 0:
            print("\n" + "=" * 60)
            resp = input("Download AI models now? (Recommended) [Y/n]: ")
            if resp.strip().lower() not in ("n", "no"):
                download_all_models(args.whisper_size)
        
        return res.returncode

    # Show info only (when --no-install is used)
    print("Detected profiles:")
    print(" - core: %d packages" % len(core))
    for k, v in extras.items():
        print(f" - {k}: {len(v)} packages")
    print("\nUsage:")
    print("  python set_dependencies.py                    # Install all packages + download models")
    print("  python set_dependencies.py --no-install       # Show info only")
    print("  python set_dependencies.py --download-models  # Download AI models only")
    print("  python set_dependencies.py --whisper-size small  # Use different Whisper model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
