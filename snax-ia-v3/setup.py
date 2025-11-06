from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="snax-ia-v3",
    version="3.0.0",
    author="Seu Nome",
    author_email="seu.email@exemplo.com",
    description="Modelo de linguagem compacto otimizado para mobile",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seu-usuario/snax-ia-v3",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "sentencepiece>=0.1.99",
        "datasets>=2.14.0",
        "transformers>=4.30.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
        "onnx>=1.14.0",
        "onnxruntime>=1.15.0"
    ],
    entry_points={
        "console_scripts": [
            "snax-train=train_v3:main",
            "snax-chat=chat_v3:main",
            "snax-export=export_mobile:main"
        ],
    },
)