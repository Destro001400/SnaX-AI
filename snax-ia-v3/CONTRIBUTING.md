# 🤝 Guia de Contribuição - SnaX IA v3

Obrigado por considerar contribuir com o projeto SnaX IA v3! Este documento fornece diretrizes para contribuir de forma efetiva.

## 🐛 Reportando Bugs

1. Use o template de bug report
2. Verifique se o bug já não foi reportado
3. Inclua:
   - Passos para reproduzir
   - Comportamento esperado vs observado
   - Versões (Python, PyTorch, etc)
   - Stack trace completo
   - Snippets de código mínimos

## 💡 Sugerindo Features

1. Use o template de feature request
2. Descreva claramente o problema/necessidade
3. Explique a solução que você gostaria
4. Considere alternativas
5. Forneça exemplos de uso

## 🔄 Pull Requests

### Preparação

1. Fork o repositório
2. Clone seu fork:
```bash
git clone https://github.com/seu-usuario/snax-ia-v3
cd snax-ia-v3
```

3. Crie uma branch:
```bash
git checkout -b feature/nome-da-feature
```

### Desenvolvimento

1. Siga os padrões de código
2. Adicione testes para novas features
3. Atualize a documentação
4. Mantenha commits atômicos

### Submissão

1. Push para seu fork:
```bash
git push origin feature/nome-da-feature
```

2. Abra um Pull Request:
   - Use o template fornecido
   - Vincule issues relacionadas
   - Descreva as mudanças
   - Adicione screenshots se relevante

## 📝 Padrões de Código

### Python

1. Siga PEP 8
2. Use type hints
3. Docstrings em todas as funções/classes
4. Máximo de 88 caracteres por linha
5. Imports organizados:
```python
# Stdlib
import os
from typing import List

# Third party
import torch
import numpy as np

# Local
from model_v3 import SnaXIA_v3
```

### Docstrings

Use o formato Google:
```python
def function(arg1: int, arg2: str) -> bool:
    """Breve descrição.

    Descrição mais longa se necessário.

    Args:
        arg1: Descrição do arg1
        arg2: Descrição do arg2

    Returns:
        Descrição do retorno

    Raises:
        ValueError: Quando arg1 < 0
    """
```

### Testes

1. Use pytest
2. Nomeie testes descritivamente
3. Uma assertion por teste
4. Use fixtures quando possível
5. Organize em classes por funcionalidade

## 📦 Estrutura de Commits

Use commits semânticos:

- `feat`: Nova feature
- `fix`: Correção de bug
- `docs`: Documentação
- `style`: Formatação
- `refactor`: Refatoração
- `test`: Testes
- `chore`: Manutenção

Exemplos:
```
feat(model): adiciona suporte a RoPE
fix(train): corrige memory leak no dataloader
docs(readme): atualiza benchmarks
```

## 🚀 Release

1. Atualize CHANGELOG.md
2. Bump versão em setup.py
3. Crie tag git
4. Push para PyPI

## ⚖️ Licença

Ao contribuir, você concorda que suas contribuições estarão sob a mesma licença Apache 2.0 do projeto.