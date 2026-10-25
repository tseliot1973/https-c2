"""
setup.py — build the tcp_sessions_ext C++ extension.

Build in-place:
    python3 setup.py build_ext --inplace

Requires:
    pip3 install nanobind
    libpcap (already installed via Homebrew or Xcode CommandLineTools)
"""

import subprocess
from pathlib import Path
from setuptools import setup, Extension

import nanobind

nanobind_dir = Path(nanobind.__file__).parent

# macOS SDK path (provides pcap.h and system headers)
try:
    sdk = subprocess.check_output(
        ["xcrun", "--sdk", "macosx", "--show-sdk-path"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    # Avoid injecting SDK C include paths as -I entries. Doing so can
    # reorder libc++/C headers and break standard header resolution.
    extra_includes = []
    extra_link_args = [f"-isysroot{sdk}"]
except Exception:
    extra_includes = []
    extra_link_args = []

ext = Extension(
    name="tcp_sessions_ext",
    sources=[
        "tcp_sessions_ext.cpp",
        str(nanobind_dir / "src" / "nb_combined.cpp"),
    ],
    include_dirs=[
        nanobind.include_dir(),
        str(nanobind_dir / "ext" / "robin_map" / "include"),
        "/usr/local/include",
    ] + extra_includes,
    library_dirs=["/usr/local/lib"],
    libraries=["pcap"],
    define_macros=[("NB_COMPACT_ASSERTIONS", "1")],
    extra_compile_args=["-std=c++17", "-O2", "-Wall", "-fvisibility=hidden"],
    extra_link_args=extra_link_args,
    language="c++",
)

setup(
    name="tcp_sessions_ext",
    version="1.0.0",
    description="C++ TCP session extractor with nanobind Python bindings",
    ext_modules=[ext],
)
