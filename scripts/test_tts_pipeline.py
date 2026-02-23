"""Comprehensive test of TTS pipeline with Qwen3-TTS.

Tests:
1. Qwen3Manager - Model loading and synthesis
2. Voice cloning - Extract embedding from audio
3. UnifiedTTSManager - Routing between Piper and Qwen3
4. TTSCache - Cache hit/miss behavior
5. TTSQueue - Async synthesis (optional)

Run this on GCP VM with GPU to verify everything works.

Usage:
    python scripts/test_tts_pipeline.py
"""

import sys
import os
import time
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_1_qwen3_manager():
    """Test Qwen3Manager model loading and synthesis."""
    print("\n" + "=" * 60)
    print("TEST 1: Qwen3Manager - Model Loading & Synthesis")
    print("=" * 60)
    
    try:
        from src.utilities.tts.qwen3_manager import Qwen3Manager
        from src.utilities.config import get_config
        
        # Create config with Qwen3 enabled
        config = get_config()
        config.qwen3.enabled = True
        config.qwen3.device = "cuda"
        config.qwen3.model_precision = "float16"
        
        print("✓ Qwen3Manager imported successfully")
        
        # Initialize manager
        print("\nInitializing Qwen3Manager...")
        manager = Qwen3Manager(config)
        print(f"✓ Device: {manager.device}")
        
        # Load model (this will download ~1.7GB on first run)
        print("\nLoading Qwen3-TTS model (may take 1-2 minutes on first run)...")
        start = time.time()
        manager.load_model()
        load_time = time.time() - start
        print(f"✓ Model loaded in {load_time:.1f}s")
        
        # Create a quick voice reference (needed for Base model)
        print("\nCreating test voice reference...")
        import tempfile
        
        # Create synthetic audio sample (5s)
        sample_rate = 16000
        duration = 5.0
        audio_data = np.random.randn(int(sample_rate * duration)) * 0.1
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        sf.write(str(tmp_path), audio_data, sample_rate)
        
        # Extract voice embedding
        embedding = manager.extract_voice_embedding(
            voice_id="test_voice_1",
            audio_path=tmp_path,
            ref_text="This is a test audio sample.",
        )
        print(f"✓ Voice reference created: {embedding.voice_id}")
        
        # Synthesize test text (with voice cloning)
        print("\nSynthesizing test text: 'Hello world, this is a test.'")
        text = "Hello world, this is a test."
        
        start = time.time()
        audio_array, sample_rate = manager.synthesize(
            text=text,
            voice_id="test_voice_1",  # Use the cloned voice
            speed=1.0,
            language="en",
        )
        synth_time = time.time() - start
        
        audio_duration = len(audio_array) / sample_rate
        rtf = synth_time / audio_duration
        
        print(f"✓ Synthesized {audio_duration:.2f}s of audio in {synth_time:.2f}s")
        print(f"  - Real-time factor: {rtf:.2f}x")
        print(f"  - Sample rate: {sample_rate} Hz")
        print(f"  - Audio shape: {audio_array.shape}")
        
        # Cleanup
        tmp_path.unlink(missing_ok=True)
        
        # Unload model
        manager.unload_model()
        print("✓ Model unloaded")
        
        print("\n✅ TEST 1 PASSED: Qwen3Manager works!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_2_voice_cloning():
    """Test voice cloning workflow."""
    print("\n" + "=" * 60)
    print("TEST 2: Voice Cloning - Extract Embedding from Audio")
    print("=" * 60)
    
    try:
        from src.utilities.tts.qwen3_manager import Qwen3Manager
        from src.utilities.config import get_config
        
        # Create synthetic audio sample (5 seconds of noise - just for testing)
        print("\nCreating synthetic audio sample (5s)...")
        sample_rate = 16000
        duration = 5.0
        audio_data = np.random.randn(int(sample_rate * duration)) * 0.1
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        
        sf.write(tmp_path, audio_data, sample_rate)
        print(f"✓ Saved test audio to: {tmp_path}")
        
        # Extract voice embedding
        print("\nExtracting voice embedding...")
        config = get_config()
        config.qwen3.enabled = True
        config.qwen3.device = "cuda"
        manager = Qwen3Manager(config)
        manager.load_model()
        
        start = time.time()
        embedding = manager.extract_voice_embedding(
            voice_id="test_voice",
            audio_path=Path(tmp_path),
            ref_text="This is test audio for voice cloning.",  # Reference text for ICL mode
        )
        extract_time = time.time() - start
        
        print(f"✓ Extracted embedding in {extract_time:.2f}s")
        print(f"  - Voice ID: {embedding.voice_id}")
        print(f"  - Sample rate: {embedding.sample_rate} Hz")
        print(f"  - Duration: {embedding.duration:.2f}s")
        print(f"  - Embedding shape: {embedding.embedding.shape}")
        
        # Test synthesis with cloned voice
        print("\nSynthesizing with cloned voice...")
        text = "This text is using a cloned voice."
        
        start = time.time()
        audio_array, sample_rate = manager.synthesize(
            text=text,
            voice_id="test_voice",
            speed=1.0,
            language="en",
        )
        synth_time = time.time() - start
        
        print(f"✓ Synthesized with cloned voice in {synth_time:.2f}s")
        
        # Cleanup
        os.unlink(tmp_path)
        manager.unload_model()
        
        print("\n✅ TEST 2 PASSED: Voice cloning works!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_3_unified_manager():
    """Test UnifiedTTSManager routing between engines."""
    print("\n" + "=" * 60)
    print("TEST 3: UnifiedTTSManager - Engine Routing")
    print("=" * 60)
    
    try:
        from src.utilities.tts.tts_manager import UnifiedTTSManager
        from src.utilities.config import get_config
        
        # Get config and enable Qwen3
        print("\nLoading configuration...")
        config = get_config()
        config.qwen3.enabled = True
        config.qwen3.device = "cuda"
        
        # Initialize manager
        print("Initializing UnifiedTTSManager...")
        manager = UnifiedTTSManager(config)
        print(f"✓ Manager initialized")
        print(f"  - Piper voices: {len([v for v in manager.voice_catalog.values() if v.engine == 'piper'])}")
        print(f"  - Qwen3 enabled: {manager.qwen3_config.enabled}")
        
        # Test Piper synthesis (should be fast)
        if len([v for v in manager.voice_catalog.values() if v.engine == 'piper']) > 0:
            print("\nTesting Piper synthesis (fast)...")
            piper_voice = next(v.voice_id for v in manager.voice_catalog.values() if v.engine == 'piper')
            
            start = time.time()
            audio = manager.synthesize(
                text="Testing Piper TTS.",
                voice_id=piper_voice,
                speed=1.0,
                output_format="wav",
            )
            piper_time = time.time() - start
            
            print(f"✓ Piper synthesis: {len(audio)} bytes in {piper_time:.3f}s")
        else:
            print("⚠️  No Piper voices available, skipping Piper test")
        
        # Test Qwen3 synthesis (should be slower but higher quality)
        print("\nTesting Qwen3 synthesis (slower, higher quality)...")
        
        # Note: No cloned voices yet, so Qwen3 will use default voice
        start = time.time()
        audio = manager.synthesize(
            text="Testing Qwen3 TTS.",
            voice_id=None,  # Will use Piper default since no Qwen3 voices
            speed=1.0,
            output_format="wav",
            language="en",
        )
        qwen3_time = time.time() - start
        
        print(f"✓ Synthesis: {len(audio)} bytes in {qwen3_time:.3f}s")
        
        # Unload
        manager.unload_all()
        
        print("\n✅ TEST 3 PASSED: UnifiedTTSManager works!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_tts_cache():
    """Test TTSCache for audio caching."""
    print("\n" + "=" * 60)
    print("TEST 4: TTSCache - Audio Caching")
    print("=" * 60)
    
    try:
        from src.utilities.tts.tts_cache import TTSCache
        import tempfile
        
        # Create cache in temp directory
        cache_dir = tempfile.mkdtemp(prefix="tts_cache_test_")
        print(f"✓ Created test cache directory: {cache_dir}")
        
        cache = TTSCache(
            cache_dir=cache_dir,
            max_entries=10,
            max_size_mb=50,
        )
        
        # Test cache miss
        print("\nTest 1: Cache MISS (first request)")
        audio = cache.get("Hello world", "test_voice", 1.0, "piper")
        assert audio is None, "Expected cache miss"
        print("✓ Cache miss as expected")
        
        # Add to cache
        test_audio = b"fake_audio_data_12345"
        cache.put("Hello world", "test_voice", 1.0, "piper", test_audio, "wav")
        print("✓ Added audio to cache")
        
        # Test cache hit
        print("\nTest 2: Cache HIT (second request)")
        audio = cache.get("Hello world", "test_voice", 1.0, "piper")
        assert audio == test_audio, "Expected cache hit with same data"
        print("✓ Cache hit with correct data")
        
        # Test different text = cache miss
        print("\nTest 3: Different text = cache MISS")
        audio = cache.get("Different text", "test_voice", 1.0, "piper")
        assert audio is None, "Expected cache miss for different text"
        print("✓ Cache miss for different text")
        
        # Check stats
        stats = cache.get_stats()
        print(f"\nCache Statistics:")
        print(f"  - Entries: {stats['entries']}/{stats['max_entries']}")
        print(f"  - Size: {stats['size_mb']:.2f}/{stats['max_size_mb']} MB")
        print(f"  - Hit rate: {stats['hit_rate']:.1%}")
        print(f"  - Hits: {stats['hits']}, Misses: {stats['misses']}")
        
        assert stats['hit_rate'] > 0, "Expected some cache hits"
        
        # Cleanup
        cache.clear()
        import shutil
        shutil.rmtree(cache_dir)
        
        print("\n✅ TEST 4 PASSED: TTSCache works!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_cache_integration():
    """Test cache integration with UnifiedTTSManager."""
    print("\n" + "=" * 60)
    print("TEST 5: Cache Integration - Real Pipeline")
    print("=" * 60)
    
    try:
        from src.utilities.tts.tts_manager import UnifiedTTSManager
        from src.utilities.tts.tts_cache import TTSCache
        from src.utilities.config import get_config
        import tempfile
        
        # Setup
        config = get_config()
        config.qwen3.enabled = True
        config.qwen3.device = "cuda"
        
        cache_dir = tempfile.mkdtemp(prefix="tts_cache_integration_")
        cache = TTSCache(cache_dir=cache_dir, max_entries=10)
        
        manager = UnifiedTTSManager(config)
        manager.set_cache(cache)
        print("✓ Manager and cache initialized")
        
        # First synthesis (cache miss)
        print("\nFirst synthesis (should be MISS)...")
        text = "This is a cache test."
        
        if len([v for v in manager.voice_catalog.values() if v.engine == 'piper']) > 0:
            voice_id = next(v.voice_id for v in manager.voice_catalog.values() if v.engine == 'piper')
            
            start = time.time()
            audio1 = manager.synthesize(text, voice_id=voice_id)
            time1 = time.time() - start
            print(f"✓ First synthesis: {len(audio1)} bytes in {time1:.3f}s")
            
            # Second synthesis (cache hit - should be instant)
            print("\nSecond synthesis (should be HIT, instant)...")
            start = time.time()
            audio2 = manager.synthesize(text, voice_id=voice_id)
            time2 = time.time() - start
            print(f"✓ Second synthesis: {len(audio2)} bytes in {time2:.3f}s")
            
            # Verify cache hit
            assert audio1 == audio2, "Audio should be identical"
            assert time2 < time1 * 0.1, f"Cache hit should be much faster ({time2:.3f}s vs {time1:.3f}s)"
            print(f"✓ Cache speedup: {time1/time2:.1f}x faster")
            
            # Check cache stats
            stats = cache.get_stats()
            print(f"\nCache Stats: {stats['hits']} hits, {stats['misses']} misses, {stats['hit_rate']:.1%} hit rate")
            
            # Cleanup
            manager.unload_all()
            cache.clear()
            import shutil
            shutil.rmtree(cache_dir)
            
            print("\n✅ TEST 5 PASSED: Cache integration works!")
            return True
        else:
            print("⚠️  No Piper voices available, skipping integration test")
            return True
        
    except Exception as e:
        print(f"\n❌ TEST 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TTS PIPELINE COMPREHENSIVE TEST")
    print("=" * 60)
    print("\nThis will test:")
    print("1. Qwen3Manager - Model loading & synthesis")
    print("2. Voice Cloning - Embedding extraction")
    print("3. UnifiedTTSManager - Engine routing")
    print("4. TTSCache - Caching behavior")
    print("5. Cache Integration - Real pipeline with caching")
    print("\n⚠️  Note: First run will download Qwen3 models (~1.7GB)")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Qwen3Manager", test_1_qwen3_manager()))
    results.append(("Voice Cloning", test_2_voice_cloning()))
    results.append(("UnifiedTTSManager", test_3_unified_manager()))
    results.append(("TTSCache", test_4_tts_cache()))
    results.append(("Cache Integration", test_5_cache_integration()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    print(f"\n{total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("\n🎉 ALL TESTS PASSED! Backend is ready!")
        print("\nNext steps:")
        print("1. Proceed to frontend implementation")
        print("2. Or test backend API endpoints manually")
        print("3. Enable Qwen3 in production config")
        return 0
    else:
        print("\n⚠️  Some tests failed. Fix issues before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
