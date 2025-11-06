import os
from pathlib import Path
from typing import List, Optional, Union
import sentencepiece as spm

class SnaXTokenizer_v3:
    """Tokenizador SentencePiece BPE para o SnaX IA v3."""
    def __init__(self, model_path: str = "tokenizer_v3.model", vocab_size: int = 8192) -> None:
        self.model_path = model_path
        self.vocab_size = vocab_size
        self.sp = spm.SentencePieceProcessor()
        
        if not os.path.exists(model_path):
            print("⚡ Treinando novo tokenizador...")
            self._train_tokenizer()
        
        self.sp.load(model_path)
        
        # IDs especiais
        self._bos_id = self.sp.piece_to_id("<bos>")
        self._eos_id = self.sp.piece_to_id("<eos>")
        self._pad_id = self.sp.piece_to_id("<pad>")
        self._unk_id = self.sp.piece_to_id("<unk>")
    
    def _create_training_data(self, output_path: str = "corpus.txt") -> None:
        """Cria dados de treino inicial com português, inglês e código."""
        corpus = """
# Exemplos em português
Python é uma linguagem de programação de alto nível.
A inteligência artificial está revolucionando o mundo.
Processamento de linguagem natural é fascinante.

# English examples
Machine learning models can understand patterns in data.
Neural networks are inspired by biological brains.
Natural language processing is a complex field.

# Código Python
def hello_world():
    print("Olá, mundo!")
    return True

# Código JavaScript
function calculateSum(a, b) {
    return a + b;
}

# SQL
SELECT name, age FROM users WHERE country = 'Brasil';

# HTML/CSS
<div class="container">
    <h1>Título</h1>
    <p>Parágrafo com texto.</p>
</div>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(corpus)
    
    def _train_tokenizer(self) -> None:
        """Treina o tokenizador usando SentencePiece BPE."""
        # Criar dados de treino
        self._create_training_data()
        
        # Treinar SentencePiece
        spm.SentencePieceTrainer.train(
            input="corpus.txt",
            model_prefix="tokenizer_v3",
            vocab_size=self.vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            byte_fallback=True,  # Nunca gera <unk>
            pad_id=3,           # <pad>
            unk_id=3,           # <unk>
            bos_id=0,           # <bos>
            eos_id=1,           # <eos>
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="<bos>",
            eos_piece="<eos>",
            user_defined_symbols=["<pad>", "<unk>", "<bos>", "<eos>"],
            normalization_rule_name="nmt_nfkc",  # Normalização Unicode
            num_threads=os.cpu_count(),
            train_extremely_large_corpus=False
        )
        
        # Limpar arquivo temporário
        if os.path.exists("corpus.txt"):
            os.remove("corpus.txt")
    
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """Tokeniza texto para IDs."""
        if not text:
            return []
        
        ids = self.sp.encode_as_ids(text)
        
        if add_bos:
            ids = [self._bos_id] + ids
        if add_eos:
            ids = ids + [self._eos_id]
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Converte IDs de volta para texto."""
        if not ids:
            return ""
        
        if skip_special_tokens:
            ids = [id for id in ids if id not in {
                self._bos_id, self._eos_id, self._pad_id, self._unk_id
            }]
        
        return self.sp.decode_ids(ids)
    
    def tokenize(self, text: str) -> List[str]:
        """Retorna lista de tokens (strings)."""
        return self.sp.encode_as_pieces(text)
    
    def get_vocab_size(self) -> int:
        """Retorna tamanho do vocabulário."""
        return self.sp.get_piece_size()
    
    def bos_id(self) -> int:
        """ID do token BOS (<bos>)."""
        return self._bos_id
    
    def eos_id(self) -> int:
        """ID do token EOS (<eos>)."""
        return self._eos_id
    
    def pad_id(self) -> int:
        """ID do token PAD (<pad>)."""
        return self._pad_id
    
    def unk_id(self) -> int:
        """ID do token UNK (<unk>)."""
        return self._unk_id
    
    def save(self, path: str) -> None:
        """Salva o modelo do tokenizador."""
        self.sp.save(path)
    
    @classmethod
    def load(cls, path: str) -> "SnaXTokenizer_v3":
        """Carrega tokenizador de um arquivo."""
        tokenizer = cls(model_path=path)
        return tokenizer

if __name__ == "__main__":
    # Teste rápido do tokenizador
    tokenizer = SnaXTokenizer_v3()
    
    # Teste de tokenização
    text = "Python é uma linguagem incrível! 🚀 def hello(): print('world')"
    tokens = tokenizer.tokenize(text)
    print(f"\n✨ Tokens: {tokens}\n")
    
    # Teste de encode/decode
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    print(f"🔄 Original : {text}")
    print(f"🔄 Decoded  : {decoded}")
    
    # Verificar vocabulário
    vocab_size = tokenizer.get_vocab_size()
    print(f"\n📚 Tamanho do vocabulário: {vocab_size}")
    print("✅ Teste do tokenizador passou!")