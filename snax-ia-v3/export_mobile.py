import os
import time
from pathlib import Path
import torch
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

from model_v3 import SnaXIA_v3
from tokenizer_v3 import SnaXTokenizer_v3

def convert_to_onnx(
    model_path: str,
    output_path: str,
    config: dict = None
) -> None:
    """Converte modelo PyTorch para ONNX."""
    print("🔄 Convertendo modelo para ONNX...")
    
    # Carregar modelo
    config = config or {
        "vocab_size": 8192,
        "hidden_size": 384,
        "num_layers": 6,
        "num_heads": 6,
        "num_kv_heads": 2,
        "max_seq_len": 512,
    }
    
    model = SnaXIA_v3(config)
    model.load_state_dict(torch.load(model_path, map_location="cpu")["model_state_dict"])
    model.eval()
    
    # Input dummy para exportação
    batch_size = 1
    seq_len = 32
    dummy_input = torch.randint(
        0,
        config["vocab_size"],
        (batch_size, seq_len),
        dtype=torch.long
    )
    
    # Exportar para ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
        do_constant_folding=True,
        opset_version=14,
        verbose=False
    )
    
    print(f"✅ Modelo exportado para: {output_path}")

def verify_onnx(onnx_path: str) -> None:
    """Verifica se o modelo ONNX é válido."""
    print("🔍 Verificando modelo ONNX...")
    
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✅ Modelo ONNX é válido!")
    except Exception as e:
        print(f"❌ Erro ao verificar modelo ONNX: {e}")
        raise

def quantize_onnx(
    input_path: str,
    output_path: str,
    quantize_type: QuantType = QuantType.QInt8
) -> None:
    """Quantiza modelo ONNX para INT8."""
    print("🔄 Quantizando modelo para INT8...")
    
    try:
        quantize_dynamic(
            input_path,
            output_path,
            weight_type=quantize_type,
            optimize_model=True,
            per_channel=False,
            reduce_range=False,
            nodes_to_quantize=None,
            nodes_to_exclude=None
        )
        print(f"✅ Modelo quantizado salvo em: {output_path}")
    except Exception as e:
        print(f"❌ Erro ao quantizar modelo: {e}")
        raise

def benchmark_model(
    onnx_path: str,
    tokenizer_path: str = "tokenizer_v3.model",
    num_warmup: int = 5,
    num_runs: int = 100
) -> None:
    """Benchmark do modelo ONNX."""
    print("\n📊 Iniciando benchmark...")
    
    # Carregar tokenizer e preparar input
    tokenizer = SnaXTokenizer_v3(tokenizer_path)
    text = "O modelo de linguagem SnaX IA v3"
    input_ids = tokenizer.encode(text)
    input_tensor = np.array([input_ids], dtype=np.int64)
    
    # Configurar sessão ONNX Runtime
    providers = ['CPUExecutionProvider']
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.insert(0, 'CUDAExecutionProvider')
    
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    # Warmup
    print("🔥 Realizando warmup...")
    for _ in range(num_warmup):
        session.run(None, {"input_ids": input_tensor})
    
    # Benchmark
    print(f"⚡ Executando {num_runs} inferências...")
    latencies = []
    
    for _ in range(num_runs):
        start_time = time.perf_counter()
        session.run(None, {"input_ids": input_tensor})
        latency = (time.perf_counter() - start_time) * 1000  # ms
        latencies.append(latency)
    
    # Estatísticas
    latencies = np.array(latencies)
    avg_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p90_latency = np.percentile(latencies, 90)
    p99_latency = np.percentile(latencies, 99)
    
    print("\n📈 Resultados do benchmark:")
    print(f"   Média: {avg_latency:.2f}ms")
    print(f"   P50  : {p50_latency:.2f}ms")
    print(f"   P90  : {p90_latency:.2f}ms")
    print(f"   P99  : {p99_latency:.2f}ms")
    
    # Informações do modelo
    model_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"\n📦 Tamanho do modelo: {model_size_mb:.2f}MB")
    print(f"🖥️  Providers: {session.get_providers()}")

def main():
    """Função principal de exportação."""
    # Caminhos
    model_dir = Path("checkpoints")
    onnx_dir = Path("onnx")
    onnx_dir.mkdir(exist_ok=True)
    
    # Encontrar melhor checkpoint
    best_checkpoint = model_dir / "best_checkpoint.pt"
    if not best_checkpoint.exists():
        checkpoints = list(model_dir.glob("step_*.pt"))
        if not checkpoints:
            raise FileNotFoundError("Nenhum checkpoint encontrado!")
        best_checkpoint = max(checkpoints, key=os.path.getctime)
    
    # Nomes dos arquivos
    base_onnx = onnx_dir / "snaxia_v3.onnx"
    quantized_onnx = onnx_dir / "snaxia_v3_int8.onnx"
    
    # Pipeline de exportação
    try:
        # 1. Converter para ONNX
        convert_to_onnx(str(best_checkpoint), str(base_onnx))
        
        # 2. Verificar modelo
        verify_onnx(str(base_onnx))
        
        # 3. Quantizar para INT8
        quantize_onnx(str(base_onnx), str(quantized_onnx))
        
        # 4. Benchmark
        print("\n🔍 Benchmark do modelo FP32:")
        benchmark_model(str(base_onnx))
        
        print("\n🔍 Benchmark do modelo INT8:")
        benchmark_model(str(quantized_onnx))
        
        print("\n✨ Exportação concluída com sucesso!")
        
    except Exception as e:
        print(f"\n❌ Erro durante exportação: {e}")
        raise

if __name__ == "__main__":
    main()