"""
Cliente para API de Ollama (LLM local).
"""

import requests
from typing import Optional


class OllamaClient:
    def __init__(self, host: str = "localhost", port: int = 11434):
        self.base_url = f"http://{host}:{port}"
    
    def check_health(self) -> bool:
        """Verifica que Ollama esté corriendo"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate(self, prompt: str, model: str = "huihui_ai/qwen3-abliterated:8b-v2",
                 system: str = None, temperature: float = 0.7) -> Optional[str]:
        """
        Genera texto con Ollama.
        
        Args:
            prompt: Prompt del usuario
            model: Modelo a usar
            system: Prompt del sistema
            temperature: Creatividad (0-1)
            
        Returns:
            Texto generado
        """
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature}
            }
            
            if system:
                payload["system"] = system
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('response')
            return None
            
        except Exception as e:
            print(f"Error generating with Ollama: {e}")
            return None


class OllamaPromptEnhancer:
    """Mejora prompts para generación de imágenes"""
    
    SYSTEM_PROMPT = """Eres un experto en prompts para Stable Diffusion.
    
Tu tarea es expandir prompts simples en descripciones detalladas y efectivas.
Incluí: sujeto, estilo, iluminación, composición, calidad técnica.
Respondé SOLO el prompt mejorado, sin explicaciones."""
    
    def __init__(self, model: str = "huihui_ai/qwen3-abliterated:8b-v2"):
        self.client = OllamaClient()
        self.model = model
    
    def expandir(self, prompt_simple: str) -> str:
        """
        Expande prompt simple a uno detallado.
        
        Ejemplo:
            "un gato" → "a detailed photograph of a fluffy orange cat..."
        """
        resultado = self.client.generate(
            prompt=f"Mejorá este prompt para SDXL: {prompt_simple}",
            model=self.model,
            system=self.SYSTEM_PROMPT,
            temperature=0.7
        )
        
        return resultado.strip() if resultado else prompt_simple
    
    def traducir(self, prompt: str, idioma_destino: str = "en") -> str:
        """Traduce prompt a inglés para mejor resultado en SD"""
        if idioma_destino == "en":
            instruccion = "Traducí al inglés este prompt para Stable Diffusion:"
        else:
            instruccion = f"Traducí a {idioma_destino}:"
        
        resultado = self.client.generate(
            prompt=f"{instruccion} {prompt}",
            model=self.model,
            temperature=0.3
        )
        
        return resultado.strip() if resultado else prompt
