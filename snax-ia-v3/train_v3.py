import os
import math
import time
from pathlib import Path
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset
from tqdm.auto import tqdm

from model_v3 import SnaXIA_v3
from tokenizer_v3 import SnaXTokenizer_v3

class TextDataset(Dataset):
    """Dataset para treinamento do modelo."""
    def __init__(self, texts: list, tokenizer: SnaXTokenizer_v3,
                 max_length: int = 512) -> None:
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        text = self.texts[idx]
        ids = self.tokenizer.encode(text)
        
        if len(ids) > self.max_length:
            start = torch.randint(0, len(ids) - self.max_length, (1,)).item()
            ids = ids[start:start + self.max_length]
        else:
            ids = ids + [self.tokenizer.pad_id()] * (self.max_length - len(ids))
        
        return torch.tensor(ids)

class Trainer:
    """Classe para treinar o modelo SnaX IA v3."""
    def __init__(
        self,
        model: nn.Module,
        tokenizer: SnaXTokenizer_v3,
        config: dict = None
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or {
            "batch_size": 32,
            "grad_accum_steps": 4,
            "learning_rate": 3e-4,
            "warmup_steps": 1000,
            "max_steps": 20000,
            "eval_interval": 500,
            "save_interval": 1000,
            "max_length": 512,
            "mixed_precision": True,
            "checkpoint_dir": "checkpoints",
        }
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Otimizador e scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )
        
        # Learning rate scheduler com warmup
        self.scheduler = self.get_lr_scheduler()
        
        # Gradient scaler para mixed precision
        self.scaler = GradScaler(enabled=self.config["mixed_precision"])
        
        # Criar diretório para checkpoints
        os.makedirs(self.config["checkpoint_dir"], exist_ok=True)
    
    def get_lr_scheduler(self):
        """Cria learning rate scheduler com warmup."""
        def lr_lambda(step):
            if step < self.config["warmup_steps"]:
                return float(step) / float(max(1, self.config["warmup_steps"]))
            return 0.5 * (1.0 + math.cos(
                math.pi * (step - self.config["warmup_steps"]) /
                (self.config["max_steps"] - self.config["warmup_steps"])
            ))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def save_checkpoint(self, step: int, loss: float, best: bool = False) -> None:
        """Salva checkpoint do modelo."""
        checkpoint = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": loss,
            "config": self.config,
        }
        
        prefix = "best" if best else f"step_{step}"
        path = Path(self.config["checkpoint_dir"]) / f"{prefix}_checkpoint.pt"
        torch.save(checkpoint, path)
        print(f"💾 Checkpoint salvo em: {path}")
    
    def load_checkpoint(self, path: str) -> int:
        """Carrega checkpoint do modelo."""
        print(f"📂 Carregando checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        return checkpoint["step"]
    
    def train_step(
        self,
        batch: torch.Tensor,
        grad_accum: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Executa um passo de treinamento."""
        # Shift para criar input e target
        x = batch[:, :-1]
        y = batch[:, 1:]
        
        # Forward pass com mixed precision
        with autocast(enabled=self.config["mixed_precision"]):
            logits = self.model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=self.tokenizer.pad_id()
            )
            
            # Gradient accumulation
            if grad_accum:
                loss = loss / self.config["grad_accum_steps"]
        
        # Backward pass com mixed precision
        self.scaler.scale(loss).backward()
        
        return loss, logits
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> float:
        """Avalia o modelo no conjunto de validação."""
        self.model.eval()
        total_loss = 0
        total_steps = 0
        
        for batch in tqdm(val_loader, desc="🔍 Avaliando", leave=False):
            batch = batch.to(self.device)
            x = batch[:, :-1]
            y = batch[:, 1:]
            
            with autocast(enabled=self.config["mixed_precision"]):
                logits = self.model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=self.tokenizer.pad_id()
                )
            
            total_loss += loss.item()
            total_steps += 1
        
        self.model.train()
        return total_loss / total_steps
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        initial_step: int = 0
    ) -> None:
        """Treina o modelo."""
        print(f"🚀 Iniciando treinamento no dispositivo: {self.device}")
        self.model.train()
        
        best_val_loss = float("inf")
        train_losses = []
        
        step = initial_step
        while step < self.config["max_steps"]:
            # Loop de treinamento
            for batch_idx, batch in enumerate(train_loader):
                if step >= self.config["max_steps"]:
                    break
                
                batch = batch.to(self.device)
                
                # Gradient accumulation
                is_last_accum = (
                    batch_idx + 1) % self.config["grad_accum_steps"] == 0
                
                loss, _ = self.train_step(batch, grad_accum=True)
                train_losses.append(loss.item())
                
                # Atualizar pesos após acumular gradientes
                if is_last_accum:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=1.0
                    )
                    
                    # Optimizer step com mixed precision
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    # Learning rate step
                    self.scheduler.step()
                    step += 1
                    
                    # Logging
                    if step % 100 == 0:
                        avg_loss = sum(train_losses[-100:]) / len(train_losses[-100:])
                        lr = self.scheduler.get_last_lr()[0]
                        print(f"Step {step}: loss = {avg_loss:.4f}, lr = {lr:.2e}")
                    
                    # Avaliação
                    if val_loader and step % self.config["eval_interval"] == 0:
                        val_loss = self.evaluate(val_loader)
                        print(f"📊 Validação step {step}: loss = {val_loss:.4f}")
                        
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            self.save_checkpoint(step, val_loss, best=True)
                    
                    # Salvar checkpoint regular
                    if step % self.config["save_interval"] == 0:
                        self.save_checkpoint(step, loss.item())
        
        print("✅ Treinamento concluído!")
        self.save_checkpoint(step, loss.item(), best=False)

def main():
    """Função principal de treinamento."""
    # Configurações
    config = {
        "batch_size": 32,
        "grad_accum_steps": 4,
        "learning_rate": 3e-4,
        "warmup_steps": 1000,
        "max_steps": 20000,
        "eval_interval": 500,
        "save_interval": 1000,
        "max_length": 512,
        "mixed_precision": True,
        "checkpoint_dir": "checkpoints",
    }
    
    # Carregar ou criar tokenizer
    tokenizer = SnaXTokenizer_v3()
    
    # Carregar dataset (tenta Wikipedia; se falhar, usa um corpus de exemplo)
    print("📚 Carregando dataset...")
    try:
        dataset = load_dataset("wikipedia", "20220301.pt", split="train")
        train_size = int(0.95 * len(dataset))
        train_texts = dataset[:train_size]["text"]
        val_texts = dataset[train_size:]["text"]
    except Exception as e:
        print(f"⚠️ Não foi possível carregar dataset externo ({e}). Usando corpus de exemplo pequeno para testes.")
        sample_texts = [
            "Python é uma linguagem de programação de alto nível.",
            "A inteligência artificial está revolucionando o mundo.",
            "Machine learning models learn from data.",
            "def hello():\n    print('world')",
            "Natural language processing é fascinante.",
            "A capital do Brasil é Brasília.",
            "2 + 2 = 4",
        ]
        # Repetir para criar um conjunto de tamanho razoável para testes locais
        replicated = sample_texts * 200
        split = int(0.95 * len(replicated))
        train_texts = replicated[:split]
        val_texts = replicated[split:]
    
    train_dataset = TextDataset(train_texts, tokenizer, config["max_length"])
    val_dataset = TextDataset(val_texts, tokenizer, config["max_length"])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True
    )
    
    # Criar modelo
    model = SnaXIA_v3()
    
    # Criar trainer
    trainer = Trainer(model, tokenizer, config)
    
    # Carregar checkpoint se existir
    last_checkpoint = None
    checkpoint_dir = Path(config["checkpoint_dir"])
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("step_*.pt"))
        if checkpoints:
            last_checkpoint = str(max(checkpoints, key=os.path.getctime))
    
    initial_step = 0
    if last_checkpoint:
        initial_step = trainer.load_checkpoint(last_checkpoint)
    
    # Treinar
    trainer.train(train_loader, val_loader, initial_step)

if __name__ == "__main__":
    main()