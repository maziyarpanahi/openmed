"""Utility functions for OpenMed."""

import os
from typing import Any

from .deprecation import deprecated
from .gateway import (
    GatewayLimits,
    InputValidationError,
    get_default_limits,
    normalize_text,
    validate_language,
)
from .logging import get_logger, setup_logging
from .profiling import (
    BatchMetrics,
    InferenceMetrics,
    PeakRSSMeasurement,
    Profiler,
    ProfileReport,
    Timer,
    TimingResult,
    disable_profiling,
    enable_profiling,
    get_peak_rss_bytes,
    get_profile_report,
    get_profiler,
    measure_peak_rss,
    profile,
    timed,
)
from .validation import validate_input, validate_model_name

__all__ = [
    "deprecated",
    "setup_logging",
    "get_logger",
    "validate_input",
    "validate_model_name",
    # Shared input gateway
    "InputValidationError",
    "GatewayLimits",
    "get_default_limits",
    "normalize_text",
    "validate_language",
    # Profiling utilities
    "Profiler",
    "ProfileReport",
    "TimingResult",
    "InferenceMetrics",
    "PeakRSSMeasurement",
    "BatchMetrics",
    "Timer",
    "get_profiler",
    "enable_profiling",
    "get_peak_rss_bytes",
    "disable_profiling",
    "get_profile_report",
    "measure_peak_rss",
    "profile",
    "timed",
]


def _windows_output(path: Any) -> int:
    """Create a new file with a protected owner-only DACL before any payload write."""
    import ctypes
    import msvcrt
    from ctypes import wintypes

    class SecurityAttributes(ctypes.Structure):
        _fields_ = [
            ("length", wintypes.DWORD),
            ("descriptor", wintypes.LPVOID),
            ("inherit", wintypes.BOOL),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    security = ctypes.WinDLL("advapi32", use_last_error=True)
    convert = security.ConvertStringSecurityDescriptorToSecurityDescriptorW
    convert.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPVOID),
        wintypes.LPVOID,
    ]
    convert.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [wintypes.LPVOID]
    kernel.LocalFree.restype = wintypes.LPVOID
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(SecurityAttributes),
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    descriptor = wintypes.LPVOID()
    # Protected DACL: only OWNER RIGHTS gets full access; no inherited ACEs.
    if not convert("D:P(A;;FA;;;OW)", 1, ctypes.byref(descriptor), None):
        raise OSError()
    try:
        attributes = SecurityAttributes(
            ctypes.sizeof(SecurityAttributes), descriptor, False
        )
        # GENERIC_WRITE | READ_CONTROL, no sharing, CREATE_NEW, OPEN_REPARSE_POINT.
        handle = kernel.CreateFileW(
            os.fsdecode(path),
            0x40020000,
            0,
            ctypes.byref(attributes),
            1,
            0x00200080,
            None,
        )
        if handle == ctypes.c_void_p(-1).value:
            raise OSError()
        try:
            return msvcrt.open_osfhandle(handle, os.O_WRONLY | os.O_BINARY)
        except BaseException:
            kernel.CloseHandle(handle)
            raise
    finally:
        kernel.LocalFree(descriptor)


def _verify_windows_output(fd: int) -> None:
    """Refuse filesystems that did not preserve the requested private DACL."""
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    security = ctypes.WinDLL("advapi32", use_last_error=True)
    get_security = security.GetKernelObjectSecurity
    get_security.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    get_security.restype = wintypes.BOOL
    convert = security.ConvertSecurityDescriptorToStringSecurityDescriptorW
    convert.argtypes = [
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPWSTR),
        wintypes.LPVOID,
    ]
    convert.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [wintypes.LPVOID]
    kernel.LocalFree.restype = wintypes.LPVOID
    size = wintypes.DWORD()
    handle = msvcrt.get_osfhandle(fd)
    get_security(handle, 4, None, 0, ctypes.byref(size))
    if not 0 < size.value <= 65536:
        raise OSError()
    descriptor = ctypes.create_string_buffer(size.value)
    if not get_security(handle, 4, descriptor, size.value, ctypes.byref(size)):
        raise OSError()
    text = wintypes.LPWSTR()
    if not convert(descriptor, 1, 4, ctypes.byref(text), None):
        raise OSError()
    try:
        if text.value != "D:P(A;;FA;;;OW)":
            raise OSError()
    finally:
        kernel.LocalFree(text)
