"""Perplejidad por oración con un GPT-2 en español (familia MarIA). Se carga solo cuando se usa."""
import math
import os

NOMBRE_POR_DEFECTO = "PlanTL-GOB-ES/gpt2-base-bne"


class PerplejidadLM:
    def __init__(self, nombre=None, max_tokens=128):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.nombre = nombre or os.getenv("LM_NAME", NOMBRE_POR_DEFECTO)
        self.max_tokens = max_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(self.nombre)
        self.modelo = AutoModelForCausalLM.from_pretrained(self.nombre).to(self.device).eval()

    def perplejidades(self, oraciones):
        res = []
        inicio = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id
        for o in oraciones:
            ids = self.tok.encode(o, add_special_tokens=False)[: self.max_tokens]
            if not ids:
                continue
            x = self.torch.tensor([[inicio] + ids], device=self.device)
            with self.torch.no_grad():
                res.append(math.exp(self.modelo(x, labels=x).loss.item()))
        return res
