from setuptools import setup, find_packages

setup(
    name="aiavatar",
    version="0.8.17",
    url="https://github.com/uezo/aiavatar",
    author="uezo",
    author_email="uezo@uezo.net",
    maintainer="uezo",
    maintainer_email="uezo@uezo.net",
    description="🥰 Building AI-based conversational avatars lightning fast ⚡️💬",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["examples*", "tests*"]),
    package_data={"aiavatar": ["admin/static/*"], "voice_auth_server": ["static/*"]},
    install_requires=["httpx>=0.27.0", "openai>=1.55.3", "aiofiles>=24.1.0", "numpy>=2.2.3", "PyAudio>=0.2.14", "python-multipart>=0.0.20", "silero-vad>=6.0.0"],
    extras_require={
        "voice-auth": ["huggingface-hub[hf_xet]>=0.24.0", "mlx>=0.28.0", "torch>=2.8.0", "torchaudio>=2.8.0"],
    },
    entry_points={
        "console_scripts": [
            "server=server.server:main",
            "voice-auth-enroll=server.voice_auth_enroll:main",
            "voice-auth-server=voice_auth_server.server:main",
            "voice-auth-runtime=voice_auth_runtime.server:main",
        ],
    },
    license="Apache v2",
    classifiers=[
        "Programming Language :: Python :: 3"
    ]
)
