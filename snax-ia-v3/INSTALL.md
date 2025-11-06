# 📚 Guia de Instalação - SnaX IA v3

## 🌐 Google Colab

1. Abra o notebook `SnaX_IA_v3_Colab.ipynb` no Google Colab
2. Execute todas as células em ordem
3. O modelo será treinado na GPU do Colab (~2-3h)

## 💻 Local (Linux/Mac)

1. Requisitos do sistema:
   - Python 3.8+
   - pip
   - git
   - Opcional: CUDA Toolkit 11.7+ (GPU)

2. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/snax-ia-v3
cd snax-ia-v3
```

3. Crie um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
```

4. Instale as dependências:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

5. Verifique a instalação:
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torch; print(f'CUDA disponível: {torch.cuda.is_available()}')"
```

## 🪟 Windows

1. Requisitos:
   - Python 3.8+ (da Microsoft Store ou python.org)
   - git
   - Opcional: CUDA Toolkit 11.7+ (GPU)

2. Clone o repositório:
```powershell
git clone https://github.com/seu-usuario/snax-ia-v3
cd snax-ia-v3
```

3. Crie um ambiente virtual:
```powershell
python -m venv venv
.\venv\Scripts\activate
```

4. Instale as dependências:
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

5. Verifique a instalação:
```powershell
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torch; print(f'CUDA disponível: {torch.cuda.is_available()}')"
```

## 🐋 Docker

1. Construa a imagem:
```bash
docker build -t snax-ia-v3 .
```

2. Execute o container:
```bash
docker run -it --gpus all snax-ia-v3
```

## 🔧 Troubleshooting

### Problemas com CUDA

1. Verifique a versão do CUDA:
```bash
nvidia-smi  # Versão do driver
nvcc -V     # Versão do CUDA Toolkit
```

2. Instale a versão correta do PyTorch:
```bash
# CUDA 11.7
pip install torch --extra-index-url https://download.pytorch.org/whl/cu117

# CUDA 11.8
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
```

### Erro de memória

1. Reduza o batch size em `train_v3.py`:
```python
config = {
    "batch_size": 8,  # Reduzido de 32
    ...
}
```

2. Use gradient accumulation:
```python
config = {
    "grad_accum_steps": 8,  # Aumentado de 4
    ...
}
```

### Import errors

1. Verifique o ambiente virtual:
```bash
# Linux/Mac
which python
pip list

# Windows
where python
pip list
```

2. Reinstale dependências:
```bash
pip uninstall -y -r requirements.txt
pip install -r requirements.txt
```

## 📞 Suporte

Se encontrar problemas:

1. Verifique as [Issues](https://github.com/seu-usuario/snax-ia-v3/issues)
2. Consulte nossa [FAQ](docs/faq.md)
3. Abra uma nova issue com:
   - Sistema operacional e versão
   - Versão do Python
   - Logs de erro completos
   - Passos para reproduzir