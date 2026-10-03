"""
Pipeline de texto a imagen (txt2img) con SDXL.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

from utils.comfyui_client import ComfyUIClient
from utils.ollama_client import OllamaPromptEnhancer


class Txt2ImgPipeline:
    def __init__(self, comfyui_host: str = "localhost", comfyui_port: int = 8188):
        self.client = ComfyUIClient(comfyui_host, comfyui_port)
        self.enhancer = OllamaPromptEnhancer()
    
    def build_workflow(self, prompt: str, negative_prompt: str = "",
                       width: int = 1024, height: int = 1024,
                       steps: int = 4, cfg: float = 1.0,
                       seed: int = -1) -> dict:
        """
        Construye workflow de ComfyUI para SDXL Turbo.
        
        Formato: API JSON de ComfyUI
        """
        workflow = {
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": "sdxl_turbo.safetensors"
                }
            },
            "5": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["4", 1]
                }
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["4", 1]
                }
            },
            "7": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["4", 0],
                    "positive": ["5", 0],
                    "negative": ["6", 0],
                    "latent_image": ["8", 0],
                    "seed": seed if seed >= 0 else int(time.time()),
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "euler_ancestral",
                    "scheduler": "normal",
                    "denoise": 1.0
                }
            },
            "8": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1
                }
            },
            "9": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["7", 0],
                    "vae": ["4", 2]
                }
            },
            "10": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["9", 0],
                    "filename_prefix": "sdxl_"
                }
            }
        }
        
        return workflow
    
    def generate(self, prompt: str, enhance_prompt: bool = True,
                 output_dir: str = "output", **kwargs) -> Optional[str]:
        """
        Genera imagen desde texto.
        
        Args:
            prompt: Descripción de la imagen
            enhance_prompt: Mejorar con Ollama
            output_dir: Carpeta de salida
            
        Returns:
            Path de la imagen generada
        """
        # Mejorar prompt si está habilitado
        if enhance_prompt:
            print(f"Prompt original: {prompt}")
            prompt = self.enhancer.expandir(prompt)
            print(f"Prompt mejorado: {prompt}")
        
        # Construir y enviar workflow
        workflow = self.build_workflow(prompt, **kwargs)
        prompt_id = self.client.queue_prompt(workflow)
        
        if not prompt_id:
            print("Error: No se pudo encolar el prompt")
            return None
        
        print(f"Prompt encolado: {prompt_id}")
        
        # Esperar resultado
        max_wait = 120  # segundos
        waited = 0
        while waited < max_wait:
            history = self.client.get_history(prompt_id)
            if prompt_id in history:
                outputs = history[prompt_id].get('outputs', {})
                if '10' in outputs:  # SaveImage node
                    images = outputs['10'].get('images', [])
                    if images:
                        filename = images[0]['filename']
                        return self.client.get_image(filename)
            
            time.sleep(2)
            waited += 2
        
        print("Timeout esperando generación")
        return None


def main():
    parser = argparse.ArgumentParser(description='Generar imagen desde texto')
    parser.add_argument('--prompt', required=True, help='Descripción de la imagen')
    parser.add_argument('--output', '-o', default='output', help='Carpeta de salida')
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--cfg', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--no-enhance', action='store_true', help='No mejorar prompt')
    
    args = parser.parse_args()
    
    pipeline = Txt2ImgPipeline()
    
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    resultado = pipeline.generate(
        prompt=args.prompt,
        enhance_prompt=not args.no_enhance,
        output_dir=args.output,
        width=args.width,
        height=args.height,
        steps=args.steps,
        cfg=args.cfg,
        seed=args.seed
    )
    
    if resultado:
        output_path = Path(args.output) / f"generated_{int(time.time())}.png"
        with open(output_path, 'wb') as f:
            f.write(resultado)
        print(f"Imagen guardada: {output_path}")
    else:
        print("Error en generación")


if __name__ == '__main__':
    main()
