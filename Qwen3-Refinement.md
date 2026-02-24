# Qwen3-TTS Performance Refinements

**Goal**: Reduce time-to-first-audio and overall synthesis time for long prompts.

**Current State** (Feb 24, 2026):
- ✅ Sentence-level chunking (~200 chars per chunk)
- ✅ Streaming endpoint (`/api/speech/synthesize-stream`) yields NDJSON per chunk
- ✅ Frontend queued audio player (plays chunk 1 while chunks 2+ synthesize)
- ✅ `torch.inference_mode()` in synthesis (upgraded from `no_grad()`)
- ✅ float16 precision
- ✅ FlashAttention 2 (if installed)
- ✅ Removed 1000-char limit (now 50,000 char ceiling)
- ✅ Pre-computed voice prompts (cached `VoiceClonePromptItem` per voice)
- ✅ Model warm-up on load (eliminates first-request JIT delay)
- ✅ Reduced hot-path logging (only first/last chunks logged)
- ✅ Configurable chunk size (`QWEN3_CHUNK_SIZE` env var, API: `/config/qwen3`)
- ✅ Background model loading (`load_model_async()` method)

---

## Performance Analysis

### Why Long Prompts Are Slow

Transformer inference scales **O(n²)** with sequence length due to self-attention.

| Text Length | Approx Synthesis Time (A100) | Notes |
|-------------|------------------------------|-------|
| 100 chars   | ~2-3s                        | Single sentence |
| 500 chars   | ~15-25s                      | Short paragraph |
| 1000 chars  | ~45-90s                      | Long paragraph |
| 2000 chars  | ~3-5 minutes                 | Multiple paragraphs |

**With chunking** (current): 1000 chars → 5 chunks × ~3s = ~15s total ✅

### Current Latency Breakdown (per chunk)

| Step | Time | Notes |
|------|------|-------|
| Reference audio load | ~100ms | Read WAV from disk |
| Voice embedding extraction | ~200-500ms | Re-computed every call |
| Text tokenization | ~10ms | Fast |
| Transformer generation | ~2-5s | Main bottleneck |
| Audio decoding | ~100ms | Neural codec → PCM |
| Speed adjustment (librosa) | ~50-200ms | Only if speed ≠ 1.0 |

---

## Proposed Optimizations

### Tier 1: Easy Wins (1-5 lines, high impact)

#### 1.1 Use `torch.inference_mode()` instead of `torch.no_grad()`
- **Impact**: 5-10% faster inference
- **Complexity**: 1 line change
- **Why**: `inference_mode` disables more bookkeeping than `no_grad`

```python
# Before
with torch.no_grad():
    wavs, sr = self.model.generate_voice_clone(...)

# After
with torch.inference_mode():
    wavs, sr = self.model.generate_voice_clone(...)
```

#### 1.2 Pre-compute Voice Clone Prompt
- **Impact**: ~200-500ms saved per chunk (10-20% overall)
- **Complexity**: ~20 lines
- **Why**: Currently re-loading reference audio and extracting embeddings for every chunk

The Qwen3 API provides `create_voice_clone_prompt()` which returns a `VoiceClonePromptItem` that can be reused:

```python
# On voice load (once):
self.voice_prompts[voice_id] = self.model.create_voice_clone_prompt(
    ref_audio=ref_audio_path,
    ref_text=ref_text,
    x_vector_only_mode=(ref_text is None),
)[0]

# On synthesis (per chunk):
wavs, sr = self.model.generate_voice_clone(
    text=chunk,
    language=language,
    voice_clone_prompt=self.voice_prompts[voice_id],  # Reuse!
)
```

#### 1.3 Model Warm-up
- **Impact**: ~5-10s saved on first request
- **Complexity**: 5 lines
- **Why**: First inference triggers JIT compilation / CUDA kernel warmup

```python
def load_model(self):
    ...
    # After loading model:
    logger.info("Warming up model...")
    with torch.inference_mode():
        self.model.generate_voice_clone(
            text="Hello",
            language="english",
            ref_audio=first_available_voice_sample,
        )
    logger.info("✓ Model warmed up")
```

#### 1.4 Reduce Logging in Hot Path
- **Impact**: ~1-5% faster
- **Complexity**: 2-3 line changes
- **Why**: `logger.debug()` has overhead even when debug level is off

```python
# Before (in _synthesize_single):
logger.debug(f"  chunk {i+1}/{len(chunks)} done in {time.time()-chunk_start:.1f}s")

# After: Move to INFO level for first/last only
if i == 0 or i == len(chunks) - 1:
    logger.info(f"Chunk {i+1}/{len(chunks)} done in {time.time()-chunk_start:.1f}s")
```

---

### Tier 2: Medium Effort (10-30 lines, good impact)

#### 2.1 Tune Chunk Size
- **Impact**: Could be 10-30% depending on text
- **Complexity**: Benchmarking required
- **Why**: 200 chars may not be optimal

| Chunk Size | Pros | Cons |
|------------|------|------|
| 100 chars | Very fast per chunk | More overhead, choppy |
| 150 chars | Fast, natural-ish | Slightly choppy |
| **200 chars** | Good balance (current) | Could tune |
| 250 chars | Fewer chunks | Slightly slower per chunk |
| 300 chars | Even fewer chunks | Noticeably slower per chunk |

**Action**: Add configurable `QWEN3_CHUNK_SIZE` env var, default 200.

#### 2.2 Background Model Loading
- **Impact**: ~10-30s saved on app startup (perceived)
- **Complexity**: ~20 lines
- **Why**: Model loading blocks startup; could do async

```python
import threading

def _load_model_async(self):
    """Load model in background thread."""
    self._loading_thread = threading.Thread(target=self.load_model, daemon=True)
    self._loading_thread.start()

def synthesize(self, ...):
    if self._loading_thread and self._loading_thread.is_alive():
        self._loading_thread.join()  # Wait if still loading
    ...
```

#### 2.3 Parallel Chunk Synthesis (if VRAM allows)
- **Impact**: Up to N× faster (where N = parallel chunks)
- **Complexity**: ~30 lines
- **Why**: Qwen3 supports batch inference

```python
# Instead of sequential:
for chunk in chunks:
    audio, sr = self._synthesize_single(chunk, ...)

# Batch (if VRAM > 8GB):
texts = [chunk for chunk in chunks[:3]]  # Batch of 3
wavs, sr = self.model.generate_voice_clone(text=texts, ...)
```

**Risk**: May exceed VRAM on smaller GPUs. Need dynamic batching based on available memory.

---

### Tier 3: Complex / Future (needs research)

#### 3.1 `torch.compile()` (Experimental)
- **Impact**: 20-40% faster (potentially)
- **Complexity**: 3 lines + testing
- **Risk**: May break with dynamic shapes, needs PyTorch 2.0+

```python
# After model load:
if hasattr(torch, 'compile'):
    self.model.model = torch.compile(self.model.model, mode="reduce-overhead")
```

**Status**: Needs testing with Qwen3 architecture.

#### 3.2 Native Model Streaming
- **Impact**: True streaming (97ms latency per Qwen3 docs)
- **Complexity**: High
- **Why**: Qwen3 supports "Dual-Track hybrid streaming generation"

The `non_streaming_mode=False` parameter exists but per docs:
> "Currently only simulates streaming text input when set to false, rather than enabling true streaming input or streaming generation."

**Action**: Monitor vLLM-Omni for proper streaming support.

#### 3.3 CUDA Graphs
- **Impact**: 20-50% faster for repeated same-shape inference
- **Complexity**: High (needs fixed input shapes)
- **Why**: Captures and replays CUDA kernel launches

Not practical for variable-length text without padding.

#### 3.4 KV-Cache Optimization
- **Impact**: Variable
- **Complexity**: Requires model surgery
- **Why**: Could persist attention KV cache across chunks for same voice

Probably handled internally by Qwen3 already via `voice_clone_prompt` reuse.

---

## Implementation Priority

### Phase 1 (COMPLETE ✅) — ~30% improvement

| Task | File | Est. Impact | Status |
|------|------|-------------|--------|
| 1.1 `torch.inference_mode()` | qwen3_manager.py | 5-10% | ✅ |
| 1.2 Pre-compute voice prompt | qwen3_manager.py | 10-20% | ✅ |
| 1.3 Model warm-up | qwen3_manager.py | First-request feels faster | ✅ |
| 1.4 Reduce hot-path logging | qwen3_manager.py | 1-5% | ✅ |

### Phase 2 (COMPLETE ✅) — Configuration & UX

| Task | File | Est. Impact | Status |
|------|------|-------------|--------|
| 2.1 Configurable chunk size | config.py, qwen3_manager.py, speech.py | Variable | ✅ |
| 2.2 Background model loading | qwen3_manager.py | Startup UX | ✅ |

### Phase 3 (Future) — Needs research

| Task | Blocker |
|------|---------|
| 3.1 torch.compile() | Needs testing |
| 3.2 Native streaming | Waiting for vLLM-Omni |
| 2.3 Batch synthesis | VRAM detection logic |

---

## Metrics to Track

After implementing, benchmark with:
- 100-char text (baseline)
- 500-char text (short paragraph)
- 1000-char text (long paragraph)
- 2000-char text (multiple paragraphs)

Measure:
1. **Time-to-first-audio** (TTFA) — how long until browser plays first chunk
2. **Total synthesis time** — wall clock for full text
3. **Real-time factor** (RTF) — synthesis time / audio duration

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/utilities/tts/qwen3_manager.py` | 1.1, 1.2, 1.3, 1.4, 2.2 |
| `src/utilities/config.py` | 2.1 (add QWEN3_CHUNK_SIZE) |

---

## References

- [Qwen3-TTS Official Repo](https://github.com/QwenLM/Qwen3-TTS)
- [vLLM-Omni Documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/)
- [PyTorch torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [FlashAttention 2](https://github.com/Dao-AILab/flash-attention)
