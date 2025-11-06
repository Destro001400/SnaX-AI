import os
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
import numpy as np

from model_v3 import SnaXIA_v3
from tokenizer_v3 import SnaXTokenizer_v3

class SnaXChat:
    """Interface de chat e testes para o SnaX IA v3."""
    def __init__(
        self,
        model_path: str = "checkpoints/best_checkpoint.pt",
        tokenizer_path: str = "tokenizer_v3.model",
        device: str = None,
        config: dict = None
    ) -> None:
        # Configurar device
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        
        print(f"🖥️ Usando device: {self.device}")
        
        # Carregar tokenizer
        self.tokenizer = SnaXTokenizer_v3(tokenizer_path)
        
        # Carregar modelo
        self.config = config or {
            "vocab_size": 8192,
            "hidden_size": 384,
            "num_layers": 6,
            "num_heads": 6,
            "num_kv_heads": 2,
            "max_seq_len": 512,
        }
        
        self.model = SnaXIA_v3(self.config)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        
        print("✅ Modelo carregado com sucesso!")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
        stop_on_eos: bool = True
    ) -> str:
        """Gera texto a partir de um prompt."""
        with torch.no_grad():
            # Tokenizar prompt
            input_ids = torch.tensor(
                self.tokenizer.encode(prompt)
            ).unsqueeze(0).to(self.device)
            
            generated = []
            past_kv = None
            
            for _ in range(max_tokens):
                # Truncar para max_seq_len se necessário
                if input_ids.shape[1] > self.config["max_seq_len"]:
                    input_ids = input_ids[:, -self.config["max_seq_len"]:]
                
                # Forward pass
                logits = self.model(input_ids)
                logits = logits[:, -1, :] / temperature
                
                # Top-k sampling
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Amostragem
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Parar se encontrar EOS
                if stop_on_eos and next_token.item() == self.tokenizer.eos_id():
                    break
                
                generated.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token], dim=1)
            
            return self.tokenizer.decode(generated)
    
    def chat(
        self,
        system_prompt: str = "Você é um assistente IA útil e amigável.",
        max_turns: int = None,
        temperature: float = 0.8,
        top_k: int = 40
    ) -> None:
        """Interface de chat interativa."""
        print("\n🤖 SnaX IA v3 Chat")
        print("Digite 'sair' para encerrar, '/help' para ajuda\n")
        print(f"🎯 {system_prompt}\n")
        
        history = []
        turn = 0
        
        while True:
            if max_turns and turn >= max_turns:
                print("\n✨ Limite de turnos atingido!")
                break
            
            # Input do usuário
            user_input = input("👤 Você: ").strip()
            
            if user_input.lower() == "sair":
                break
            
            if user_input.lower() == "/help":
                self._print_help()
                continue
            
            if user_input.startswith("/"):
                self._handle_command(user_input)
                continue
            
            # Preparar contexto com histórico
            context = system_prompt + "\n\n"
            for msg in history[-5:]:  # Últimas 5 mensagens
                context += msg + "\n"
            context += f"Usuário: {user_input}\nAssistente: "
            
            # Gerar resposta
            try:
                response = self.generate(
                    context,
                    temperature=temperature,
                    top_k=top_k
                )
                print(f"\n🤖 Assistente: {response}\n")
                
                # Atualizar histórico
                history.append(f"Usuário: {user_input}")
                history.append(f"Assistente: {response}")
                turn += 1
                
            except Exception as e:
                print(f"\n❌ Erro ao gerar resposta: {e}\n")
    
    def run_tests(self, test_file: str = None) -> Dict[str, Any]:
        """Executa testes automatizados."""
        if test_file and os.path.exists(test_file):
            with open(test_file) as f:
                tests = json.load(f)
        else:
            # Testes padrão
            tests = {
                "basic": [
                    {
                        "prompt": "A capital do Brasil é",
                        "expected_contains": ["Brasília", "Brasil"],
                    },
                    {
                        "prompt": "2 + 2 =",
                        "expected_contains": ["4", "quatro"],
                    },
                    {
                        "prompt": "Python é uma linguagem",
                        "expected_contains": ["programação"],
                    }
                ],
                "code": [
                    {
                        "prompt": "def soma(a, b):",
                        "expected_contains": ["return", "+"],
                    },
                    {
                        "prompt": "# Função para calcular fatorial",
                        "expected_contains": ["def", "return"],
                    }
                ]
            }
        
        results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "details": []
        }
        
        print("\n🔍 Iniciando testes automatizados...")
        
        for category, test_cases in tests.items():
            print(f"\n📋 Categoria: {category}")
            
            for test in test_cases:
                results["total"] += 1
                prompt = test["prompt"]
                expected = test["expected_contains"]
                
                try:
                    # Gerar resposta
                    response = self.generate(prompt, temperature=0.8)
                    
                    # Verificar expected_contains
                    passed = any(exp.lower() in response.lower() for exp in expected)
                    
                    if passed:
                        results["passed"] += 1
                        status = "✅"
                    else:
                        results["failed"] += 1
                        status = "❌"
                    
                    # Guardar detalhes
                    results["details"].append({
                        "category": category,
                        "prompt": prompt,
                        "response": response,
                        "expected": expected,
                        "passed": passed
                    })
                    
                    print(f"{status} Prompt: {prompt[:40]}...")
                    
                except Exception as e:
                    results["failed"] += 1
                    print(f"❌ Erro no teste: {str(e)}")
        
        # Relatório final
        success_rate = (results["passed"] / results["total"]) * 100
        print(f"\n📊 Resultados:")
        print(f"   Total : {results['total']}")
        print(f"   Passou: {results['passed']}")
        print(f"   Falhou: {results['failed']}")
        print(f"   Taxa  : {success_rate:.1f}%")
        
        return results
    
    def benchmark(
        self,
        num_runs: int = 100,
        text_size: str = "médio"
    ) -> None:
        """Executa benchmark de performance."""
        # Textos de teste por tamanho
        texts = {
            "pequeno": "Python é legal",
            "médio": "Python é uma linguagem de programação de alto nível.",
            "grande": """Python é uma linguagem de programação de alto nível,
                interpretada de script, imperativa, orientada a objetos,
                funcional, de tipagem dinâmica e forte."""
        }
        
        text = texts.get(text_size, texts["médio"])
        print(f"\n⚡ Iniciando benchmark com texto {text_size}...")
        print(f"   Texto: {text[:50]}...")
        
        # Warmup
        for _ in range(5):
            self.generate(text, max_tokens=20)
        
        # Benchmark
        latencies = []
        tokens_per_second = []
        
        for i in range(num_runs):
            start_time = time.perf_counter()
            output = self.generate(text, max_tokens=20)
            end_time = time.perf_counter()
            
            latency = (end_time - start_time) * 1000  # ms
            tps = len(self.tokenizer.encode(output)) / (end_time - start_time)
            
            latencies.append(latency)
            tokens_per_second.append(tps)
            
            if (i + 1) % 10 == 0:
                print(f"   Progresso: {i + 1}/{num_runs}")
        
        # Estatísticas
        latencies = np.array(latencies)
        tokens_per_second = np.array(tokens_per_second)
        
        print("\n📊 Resultados:")
        print(f"   Latência média : {np.mean(latencies):.2f}ms")
        print(f"   Latência P90   : {np.percentile(latencies, 90):.2f}ms")
        print(f"   Tokens/segundo : {np.mean(tokens_per_second):.2f}")
    
    def _print_help(self) -> None:
        """Mostra ajuda dos comandos disponíveis."""
        print("\n💡 Comandos disponíveis:")
        print("   /help         - Mostra esta ajuda")
        print("   /temp VALUE   - Ajusta temperatura (0.1-2.0)")
        print("   /topk VALUE   - Ajusta top-k (1-100)")
        print("   /test         - Executa testes automatizados")
        print("   /bench        - Executa benchmark")
        print("   sair         - Encerra o chat")
    
    def _handle_command(self, cmd: str) -> None:
        """Processa comandos especiais."""
        parts = cmd.split()
        cmd_name = parts[0].lower()
        
        if cmd_name == "/temp" and len(parts) > 1:
            try:
                temp = float(parts[1])
                if 0.1 <= temp <= 2.0:
                    self.temperature = temp
                    print(f"🎯 Temperatura ajustada para {temp}")
                else:
                    print("❌ Temperatura deve estar entre 0.1 e 2.0")
            except:
                print("❌ Valor inválido")
        
        elif cmd_name == "/topk" and len(parts) > 1:
            try:
                k = int(parts[1])
                if 1 <= k <= 100:
                    self.top_k = k
                    print(f"🎯 Top-k ajustado para {k}")
                else:
                    print("❌ Top-k deve estar entre 1 e 100")
            except:
                print("❌ Valor inválido")
        
        elif cmd_name == "/test":
            self.run_tests()
        
        elif cmd_name == "/bench":
            self.benchmark()
        
        else:
            print("❌ Comando inválido. Use /help para ajuda.")

def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description="SnaX IA v3 Chat")
    parser.add_argument(
        "--model",
        default="checkpoints/best_checkpoint.pt",
        help="Caminho para o modelo"
    )
    parser.add_argument(
        "--tokenizer",
        default="tokenizer_v3.model",
        help="Caminho para o tokenizer"
    )
    parser.add_argument(
        "--mode",
        choices=["chat", "test", "bench"],
        default="chat",
        help="Modo de execução"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        help="Device para execução"
    )
    
    args = parser.parse_args()
    
    # Criar instância do chat
    chat = SnaXChat(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        device=args.device
    )
    
    # Executar modo selecionado
    if args.mode == "chat":
        chat.chat()
    elif args.mode == "test":
        chat.run_tests()
    elif args.mode == "bench":
        chat.benchmark()

if __name__ == "__main__":
    main()