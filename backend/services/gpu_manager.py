"""
GPU detection and acceleration utilities for Orion.
Provides GPU capability detection, memory management, and CUDA optimization.
"""

import logging
import subprocess
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """GPU device information"""

    device_id: str
    name: str
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    utilization_percent: int
    temperature_c: Optional[int] = None
    is_cuda_available: bool = False


@dataclass
class GPUCapabilities:
    """System GPU capabilities"""

    has_cuda: bool
    cuda_version: Optional[str]
    gpus: List[GPUInfo]
    recommended_device: Optional[str]
    total_gpu_memory_mb: int


class GPUManager:
    """Manages GPU detection and acceleration settings"""

    def __init__(self):
        self._capabilities: Optional[GPUCapabilities] = None
        self._torch_available = False
        self._check_torch_availability()

    def _check_torch_availability(self) -> None:
        """Check if PyTorch with CUDA is available"""
        try:
            import importlib.util

            torch_spec = importlib.util.find_spec("torch")

            if torch_spec is not None:
                self._torch_available = True
                logger.debug("PyTorch available for GPU detection")
            else:
                self._torch_available = False
                logger.debug("PyTorch not available, using nvidia-ml-py fallback")
        except ImportError:
            self._torch_available = False
            logger.debug("PyTorch not available, using nvidia-ml-py fallback")

    def detect_gpu_capabilities(self) -> GPUCapabilities:
        """
        Detect system GPU capabilities and CUDA availability.

        Returns:
            GPUCapabilities with detected hardware information
        """
        if self._capabilities is not None:
            return self._capabilities

        logger.info("Detecting GPU capabilities...")

        cuda_available = False
        cuda_version = None
        gpus = []

        # Method 1: Try PyTorch detection (most reliable)
        if self._torch_available:
            cuda_available, cuda_version, gpus = self._detect_with_torch()

        # Method 2: Fallback to nvidia-smi
        if not cuda_available or not gpus:
            cuda_available, cuda_version, gpus = self._detect_with_nvidia_smi()

        # Method 3: Fallback to nvidia-ml-py
        if not cuda_available or not gpus:
            cuda_available, cuda_version, gpus = self._detect_with_nvidia_ml()

        # Determine recommended device
        recommended_device = self._select_recommended_device(gpus)

        # Calculate total GPU memory
        total_memory = sum(gpu.memory_total_mb for gpu in gpus)

        self._capabilities = GPUCapabilities(
            has_cuda=cuda_available,
            cuda_version=cuda_version,
            gpus=gpus,
            recommended_device=recommended_device,
            total_gpu_memory_mb=total_memory,
        )

        logger.info(f"GPU detection complete: CUDA={cuda_available}, GPUs={len(gpus)}, Total Memory={total_memory}MB")
        return self._capabilities

    def _detect_with_torch(self) -> Tuple[bool, Optional[str], List[GPUInfo]]:
        """Detect GPUs using PyTorch"""
        try:
            import torch

            if not torch.cuda.is_available():
                return False, None, []

            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            gpus = []

            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)

                # Get memory info
                total_memory = props.total_memory // (1024 * 1024)  # Convert to MB

                try:
                    # Current memory usage (if available)
                    torch.cuda.set_device(i)
                    used_memory = torch.cuda.memory_allocated(i) // (1024 * 1024)
                    free_memory = total_memory - used_memory
                except Exception as e:
                    logger.warning(f"Failed to get GPU memory info: {e}")
                    used_memory = 0
                    free_memory = total_memory

                gpu_info = GPUInfo(
                    device_id=f"cuda:{i}",
                    name=props.name,
                    memory_total_mb=total_memory,
                    memory_used_mb=used_memory,
                    memory_free_mb=free_memory,
                    utilization_percent=0,  # PyTorch doesn't provide utilization
                    is_cuda_available=True,
                )
                gpus.append(gpu_info)

            logger.debug(f"PyTorch detected {len(gpus)} CUDA devices")
            return True, cuda_version, gpus

        except Exception as e:
            logger.debug(f"PyTorch GPU detection failed: {e}")
            return False, None, []

    def _detect_with_nvidia_smi(self) -> Tuple[bool, Optional[str], List[GPUInfo]]:
        """Detect GPUs using nvidia-smi command"""
        try:
            # Run nvidia-smi to get GPU info with timeout
            cmd = [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

            if result.returncode != 0:
                logger.debug(f"nvidia-smi failed with return code {result.returncode}")
                return False, None, []

            # Get CUDA version
            cuda_version = None
            try:
                cuda_cmd = ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
                cuda_result = subprocess.run(cuda_cmd, capture_output=True, text=True, timeout=3)
                if cuda_result.returncode == 0:
                    cuda_version = cuda_result.stdout.strip().split("\n")[0]
            except subprocess.TimeoutExpired:
                logger.debug("nvidia-smi CUDA version query timed out")
            except Exception as e:
                logger.debug(f"CUDA version detection failed: {e}")

            # Parse GPU information
            gpus = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue

                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    try:
                        index, name, total_mem, used_mem, free_mem, util = parts[:6]
                        temp = parts[6] if len(parts) > 6 else None

                        gpu_info = GPUInfo(
                            device_id=f"cuda:{index}",
                            name=name,
                            memory_total_mb=int(total_mem) if total_mem.isdigit() else 0,
                            memory_used_mb=int(used_mem) if used_mem.isdigit() else 0,
                            memory_free_mb=int(free_mem) if free_mem.isdigit() else 0,
                            utilization_percent=int(util) if util.isdigit() else 0,
                            temperature_c=int(temp) if temp and temp.isdigit() else None,
                            is_cuda_available=True,
                        )
                        gpus.append(gpu_info)
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Failed to parse GPU line '{line}': {e}")
                        continue

            logger.debug(f"nvidia-smi detected {len(gpus)} GPUs")
            return len(gpus) > 0, cuda_version, gpus

        except subprocess.TimeoutExpired:
            logger.debug("nvidia-smi command timed out")
            return False, None, []
        except FileNotFoundError:
            logger.debug("nvidia-smi command not found")
            return False, None, []
        except Exception as e:
            logger.debug(f"nvidia-smi GPU detection failed: {e}")
            return False, None, []

    def _detect_with_nvidia_ml(self) -> Tuple[bool, Optional[str], List[GPUInfo]]:
        """Detect GPUs using nvidia-ml-py library"""
        try:
            import pynvml

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            if device_count == 0:
                return False, None, []

            # Get CUDA version
            cuda_version = None
            try:
                cuda_version = pynvml.nvmlSystemGetCudaDriverVersion_v2()
                cuda_version = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"
            except Exception as e:
                logger.warning(f"Failed to get CUDA version: {e}")

            gpus = []
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                # Get device name
                name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")

                # Get memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_mem = mem_info.total // (1024 * 1024)  # Convert to MB
                used_mem = mem_info.used // (1024 * 1024)
                free_mem = mem_info.free // (1024 * 1024)

                # Get utilization
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = util.gpu
                except Exception as e:
                    logger.warning(f"Failed to get GPU utilization: {e}")
                    gpu_util = 0

                # Get temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except Exception as e:
                    logger.warning(f"Failed to get GPU temperature: {e}")
                    temp = None

                gpu_info = GPUInfo(
                    device_id=f"cuda:{i}",
                    name=name,
                    memory_total_mb=total_mem,
                    memory_used_mb=used_mem,
                    memory_free_mb=free_mem,
                    utilization_percent=gpu_util,
                    temperature_c=temp,
                    is_cuda_available=True,
                )
                gpus.append(gpu_info)

            pynvml.nvmlShutdown()
            logger.debug(f"nvidia-ml-py detected {len(gpus)} GPUs")
            return True, cuda_version, gpus

        except Exception as e:
            logger.debug(f"nvidia-ml-py GPU detection failed: {e}")
            return False, None, []

    def _select_recommended_device(self, gpus: List[GPUInfo]) -> Optional[str]:
        """Select the best GPU device for Orion workloads"""
        if not gpus:
            return None

        if len(gpus) == 1:
            return gpus[0].device_id

        # Score GPUs based on available memory and low utilization
        best_gpu = None
        best_score = -1

        for gpu in gpus:
            # Prioritize GPUs with more available memory and lower utilization
            memory_score = gpu.memory_free_mb / 1024  # GB of free memory
            util_score = max(0, 100 - gpu.utilization_percent) / 100  # Lower utilization is better

            # Combined score (weighted toward memory availability)
            score = (memory_score * 0.7) + (util_score * 0.3)

            if score > best_score:
                best_score = score
                best_gpu = gpu

        return best_gpu.device_id if best_gpu else gpus[0].device_id

    def get_optimal_memory_limit(self, device_id: Optional[str] = None) -> Optional[int]:
        """
        Calculate optimal GPU memory limit for Orion workloads.

        Args:
            device_id: Target GPU device (None = recommended device)

        Returns:
            Recommended memory limit in MB, or None if no GPU available
        """
        capabilities = self.detect_gpu_capabilities()

        if not capabilities.has_cuda or not capabilities.gpus:
            return None

        # Find target GPU
        target_gpu = None
        target_device = device_id or capabilities.recommended_device

        for gpu in capabilities.gpus:
            if gpu.device_id == target_device:
                target_gpu = gpu
                break

        if not target_gpu:
            target_gpu = capabilities.gpus[0]  # Fallback to first GPU

        # Reserve 20% of GPU memory for system and other applications
        # Use 80% for Orion workloads
        optimal_limit = int(target_gpu.memory_total_mb * 0.8)

        logger.info(
            f"Optimal GPU memory limit for {target_gpu.device_id}: {optimal_limit}MB "
            f"(80% of {target_gpu.memory_total_mb}MB total)"
        )

        return optimal_limit

    def validate_gpu_config(self, enable_gpu: bool, device_id: Optional[str], memory_limit: Optional[int]) -> Dict[str, str]:
        """
        Validate GPU configuration settings.

        Args:
            enable_gpu: Whether GPU acceleration is enabled
            device_id: Target GPU device
            memory_limit: GPU memory limit in MB

        Returns:
            Dictionary of validation warnings/errors
        """
        warnings = {}

        if not enable_gpu:
            return warnings

        capabilities = self.detect_gpu_capabilities()

        if not capabilities.has_cuda:
            warnings["gpu_acceleration"] = "GPU acceleration enabled but CUDA not available"
            return warnings

        if not capabilities.gpus:
            warnings["gpu_devices"] = "GPU acceleration enabled but no CUDA devices found"
            return warnings

        # Validate device selection
        if device_id and device_id != "auto":
            valid_devices = [gpu.device_id for gpu in capabilities.gpus]
            if device_id not in valid_devices:
                warnings["gpu_device"] = f"Device '{device_id}' not found. Available: {valid_devices}"

        # Validate memory limit
        if memory_limit:
            target_device = device_id or capabilities.recommended_device
            target_gpu = None

            for gpu in capabilities.gpus:
                if gpu.device_id == target_device:
                    target_gpu = gpu
                    break

            if target_gpu and memory_limit > target_gpu.memory_total_mb:
                warnings["gpu_memory"] = f"Memory limit {memory_limit}MB exceeds available {target_gpu.memory_total_mb}MB"

            if target_gpu and memory_limit > target_gpu.memory_free_mb:
                warnings["gpu_memory_usage"] = (
                    f"Memory limit {memory_limit}MB exceeds free memory {target_gpu.memory_free_mb}MB"
                )

        return warnings


# Global GPU manager instance
_gpu_manager: Optional[GPUManager] = None


def get_gpu_manager() -> GPUManager:
    """Get global GPU manager instance"""
    global _gpu_manager

    if _gpu_manager is None:
        _gpu_manager = GPUManager()

    return _gpu_manager


def detect_gpu_capabilities() -> GPUCapabilities:
    """Convenience function to detect GPU capabilities"""
    return get_gpu_manager().detect_gpu_capabilities()


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    gpu_manager = get_gpu_manager()

    # Detect capabilities
    capabilities = gpu_manager.detect_gpu_capabilities()

    print(f"CUDA Available: {capabilities.has_cuda}")
    print(f"CUDA Version: {capabilities.cuda_version}")
    print(f"GPU Count: {len(capabilities.gpus)}")
    print(f"Recommended Device: {capabilities.recommended_device}")
    print(f"Total GPU Memory: {capabilities.total_gpu_memory_mb}MB")

    for gpu in capabilities.gpus:
        print(f"\n{gpu.device_id}: {gpu.name}")
        print(f"  Memory: {gpu.memory_used_mb}MB/{gpu.memory_total_mb}MB ({gpu.memory_free_mb}MB free)")
        print(f"  Utilization: {gpu.utilization_percent}%")
        if gpu.temperature_c:
            print(f"  Temperature: {gpu.temperature_c}°C")

    # Test optimal memory limit
    if capabilities.has_cuda:
        optimal_limit = gpu_manager.get_optimal_memory_limit()
        print(f"\nOptimal Memory Limit: {optimal_limit}MB")

    # Test configuration validation
    warnings = gpu_manager.validate_gpu_config(enable_gpu=True, device_id="cuda:0", memory_limit=8192)

    if warnings:
        print(f"\nConfiguration Warnings: {warnings}")
    else:
        print("\nGPU Configuration: ✅ Valid")
