"""Setup script for LiveKit Arabic Turn Detector plugin"""

import os
from setuptools import setup, find_namespace_packages

here = os.path.abspath(os.path.dirname(__file__))

# Get version
about = {}
with open(os.path.join(here, "livekit", "plugins", "arabic_turn_detector", "version.py"), "r") as f:
    exec(f.read(), about)

# Get long description
with open(os.path.join(here, "README.md"), "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="livekit-plugins-arabic-turn-detector",
    version=about["__version__"],
    description="LiveKit plugin for Arabic end-of-utterance detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ahmed-Ezzat20/asr_demo",
    author="Ahmed Ezzat",
    author_email="",
    license="Apache-2.0",
    packages=find_namespace_packages(include=["livekit.*"]),
    python_requires=">=3.9",
    install_requires=[
        "livekit-agents>=0.8.0",
        "onnxruntime>=1.15.0",
        "numpy>=1.24.0",
        "transformers>=4.30.0",
        "arabert>=1.0.1",
        "pyarabic>=0.6.15",
        "psutil>=5.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="livekit agents arabic turn-detection eou nlp",
    project_urls={
        "Documentation": "https://github.com/Ahmed-Ezzat20/asr_demo",
        "Source": "https://github.com/Ahmed-Ezzat20/asr_demo",
        "Bug Reports": "https://github.com/Ahmed-Ezzat20/asr_demo/issues",
    },
)
