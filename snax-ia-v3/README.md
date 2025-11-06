# 🚀 SnaX IA v3 — IA Compacta para Mobile

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**SnaX IA v3 é um modelo de linguagem Transformer moderno e compacto, otimizado para rodar em dispositivos móveis.**

## 🎯 Características

- ~50M parâmetros (compacto e eficiente)
- 8k vocabulário (BPE) com suporte a PT-BR
- 512 tokens de contexto
- Arquitetura moderna:
  - RoPE (Rotary Position Embeddings)
  - GQA (Grouped-Query Attention)
  - SwiGLU Activation
- ~50MB após quantização INT8
- Treina em 2-3h no Google Colab

## 🚀 Quick Start

```bash
# Clonar repositório
git clone https://github.com/seu-usuario/snax-ia-v3
cd snax-ia-v3

# Instalar dependências
pip install -r requirements.txt

# Treinar modelo
python train_v3.py

# Testar chat
python chat_v3.py --mode chat

# Exportar para mobile (ONNX)
python export_mobile.py
```

## 📊 Benchmarks

| Modelo | Tamanho | Latência (ms/token) | RAM |
|--------|---------|---------------------|-----|
| FP32   | ~200 MB | ~250ms             | 1GB |
| INT8   | ~50 MB  | ~120ms             | 256MB |

## 💻 Requisitos

- Python 3.8+
- PyTorch 2.0+
- 8GB RAM (treino)
- GPU opcional, mas recomendada para treino

## 📱 Mobile

O modelo pode ser exportado para ONNX e rodado em Android/iOS:

1. Exporte para ONNX:
```bash
python export_mobile.py
```

2. Use ONNX Runtime Mobile:
- Android: [Tutorial Android](docs/android.md)
- iOS: [Tutorial iOS](docs/ios.md)

## 🧪 Testes

Execute os testes automatizados:
```bash
# Testes do modelo
python chat_v3.py --mode test

# Benchmark de performance
python chat_v3.py --mode bench
```

## 🤝 Contribuir

1. Leia nosso [guia de contribuição](CONTRIBUTING.md)
2. Fork o projeto
3. Crie uma branch (`git checkout -b feature/AmazingFeature`)
4. Commit suas mudanças (`git commit -m 'Add AmazingFeature'`)
5. Push para a branch (`git push origin feature/AmazingFeature`)
6. Abra um Pull Request

## 📄 Licença

Distribuído sob a licença Apache 2.0. Veja [`LICENSE`](LICENSE) para mais informações.

## ✨ Agradecimentos

- [Andrej Karpathy](https://github.com/karpathy) pela inspiração e tutoriais
- [EleutherAI](https://github.com/EleutherAI) pelas inovações em LLMs
- [Hugging Face](https://huggingface.co) pelas excelentes ferramentas