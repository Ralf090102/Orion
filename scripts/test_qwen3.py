"""Test Qwen3-TTS installation and GPU availability.

This script verifies:
1. PyTorch installation
2. CUDA availability and GPU info
3. Qwen3-TTS package import
4. FlashAttention 2 availability (optional)

Run this on your GCP VM to verify everything is set up correctly.

Usage:
    python scripts/test_qwen3.py
"""

import torch
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def test_qwen3_installation():
    """Test Qwen3-TTS installation and GPU availability."""
    print("=" * 60)
    print("Qwen3-TTS Installation Test")
    print("=" * 60)
    
    success = True
    
    # Check PyTorch
    print(f"\n✓ PyTorch version: {torch.__version__}")
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"{'✓' if cuda_available else '✗'} CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
        
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  - VRAM: {vram_gb:.1f} GB")
        
        # Check if VRAM is sufficient
        if vram_gb < 4.0:
            print(f"  ⚠️  WARNING: VRAM {vram_gb:.1f}GB < 4GB recommended")
            print(f"     Qwen3-TTS may be slow or fail with OOM errors")
        else:
            print(f"  ✓ VRAM sufficient for Qwen3-TTS")
    else:
        print("  ⚠️  WARNING: No CUDA GPU detected. Qwen3-TTS will be VERY slow on CPU.")
        print("     For production use, a GPU with 4+ GB VRAM is required.")
        success = False
    
    # Try importing Qwen3-TTS
    print("\nChecking Qwen3-TTS package...")
    try:
        from qwen_tts import Qwen3TTSModel
        print(f"✓ Qwen3-TTS package installed successfully")
        print(f"  - Import path: qwen_tts.Qwen3TTSModel")
        
        # Try to get model info
        print(f"\nChecking available models...")
        print(f"  - Recommended: Qwen/Qwen3-TTS-12Hz-1.7B-Base (voice cloning)")
        print(f"  - Models will auto-download on first use")
        
    except ImportError as e:
        print(f"✗ Failed to import Qwen3-TTS: {e}")
        print(f"\nInstall with: pip install qwen-tts")
        success = False
    
    # Check FlashAttention
    print("\nChecking FlashAttention 2 (optional)...")
    try:
        import flash_attn
        print(f"✓ FlashAttention 2 installed (2-3x faster inference)")
        print(f"  - Version: {flash_attn.__version__ if hasattr(flash_attn, '__version__') else 'unknown'}")
    except ImportError:
        print(f"⚠️  FlashAttention 2 not installed (optional, but recommended)")
        print(f"   Install with: pip install -U flash-attn --no-build-isolation")
        print(f"   Note: Requires NVIDIA GPU with CUDA. May take 5-10 minutes to compile.")
    
    # Summary
    print("\n" + "=" * 60)
    if success:
        print("✓ All checks passed! Ready to use Qwen3-TTS")
        print("\nNext steps:")
        print("1. Enable Qwen3 in config: config.qwen3.enabled = True")
        print("2. Or set environment variable: QWEN3_ENABLED=true")
        print("3. First synthesis will download models (~1.7GB)")
    else:
        print("✗ Some checks failed. Please fix issues above.")
        print("\nMissing components:")
        if not cuda_available:
            print("  - NVIDIA GPU with CUDA support")
        if not success:
            print("  - qwen-tts package (pip install qwen-tts)")
    print("=" * 60)
    
    return success


if __name__ == "__main__":
    success = test_qwen3_installation()
    sys.exit(0 if success else 1)
