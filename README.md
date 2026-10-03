# AI Image Generation Pipeline

Sistema automatizado de generación de imágenes con IA local usando ComfyUI + Ollama.

## Arquitectura

```
ai-image-generation-pipeline/
├── README.md
├── requirements.txt
├── config/
│   ├── settings.yaml      # Configuración general
│   └── workflows.json     # Workflows de ComfyUI
├── pipelines/
│   ├── txt2img.py         # Texto a imagen
│   ├── img2img.py         # Imagen a imagen
│   ├── inpainting.py      # Inpainting/outpainting
│   └── upscale.py         # Upscaling con IA
├── utils/
│   ├── comfyui_client.py  # Cliente API ComfyUI
│   ├── ollama_client.py   # Cliente API Ollama
│   └── image_utils.py     # Procesamiento de imágenes
├── batch/
│   ├── queue_manager.py   # Gestión de colas
│   └── batch_processor.py # Procesamiento por lotes
└── output/                # Imágenes generadas
```

## Hardware requerido

- GPU: RTX 3060 Ti 8GB (o superior)
- RAM: 16GB mínimo
- Almacenamiento: 50GB+ para modelos

## Modelos soportados

| Tipo | Modelos |
|------|---------|
| SDXL | Base, Refiner, Turbo |
| SD 1.5 | Base, Inpainting |
| Upscalers | Real-ESRGAN, SwinIR |
| LLM | Ollama (local) |

## Uso rápido

```bash
# Instalar dependencias
pip install -r requirements.txt

# Generar desde texto
python pipelines/txt2img.py --prompt "paisaje montañoso" --output output/

# Procesar lote
python batch/batch_processor.py --input prompts.txt --workflow sdxl_turbo
```

## Workflows incluidos

- **SDXL Turbo**: Generación rápida (4 steps)
- **SDXL Base + Refiner**: Máxima calidad
- **Inpainting**: Edición selectiva
- **ControlNet**: Pose/detección de bordes

## API de ComfyUI

El sistema se conecta a ComfyUI en `localhost:8188` via HTTP API.

## Integración con Ollama

Prompts mejorados automáticamente con LLM local:

```python
from utils.ollama_client import OllamaPromptEnhancer

enhancer = OllamaPromptEnhancer(model="huihui_ai/qwen3-abliterated:8b-v2")
prompt_mejorado = enhancer.expandir("un gato")
# → "a detailed photograph of a fluffy cat, studio lighting, 8k..."
```

## Notas técnicas

- Latencia: ~2-5s por imagen (SDXL Turbo)
- VRAM: Optimizado para 8GB con FP8
- Batch: Procesamiento asíncrono de colas

---

**Desarrollado por:** Franco Palombarini  
**Contacto:** fran.palombarini.dev@gmail.com  
**LinkedIn:** linkedin.com/in/franpalombarini-dev
