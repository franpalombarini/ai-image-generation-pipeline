"""
Cliente HTTP para API de ComfyUI.
"""

import json
import requests
from typing import Dict, Optional


class ComfyUIClient:
    def __init__(self, host: str = "localhost", port: int = 8188):
        self.base_url = f"http://{host}:{port}"
        self.api_url = f"{self.base_url}/api"
    
    def check_health(self) -> bool:
        """Verifica que ComfyUI esté corriendo"""
        try:
            response = requests.get(f"{self.base_url}/system_stats", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def queue_prompt(self, workflow: Dict) -> Optional[str]:
        """
        Envía workflow a cola de generación.
        
        Args:
            workflow: Dict con estructura del workflow
            
        Returns:
            prompt_id si fue exitoso
        """
        try:
            payload = {"prompt": workflow}
            response = requests.post(
                f"{self.api_url}/prompt",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('prompt_id')
            return None
            
        except Exception as e:
            print(f"Error queueing prompt: {e}")
            return None
    
    def get_history(self, prompt_id: str = None) -> Dict:
        """Obtiene historial de generación"""
        url = f"{self.api_url}/history"
        if prompt_id:
            url = f"{url}/{prompt_id}"
        
        response = requests.get(url, timeout=10)
        return response.json() if response.status_code == 200 else {}
    
    def get_image(self, filename: str, subfolder: str = "") -> bytes:
        """Descarga imagen generada"""
        url = f"{self.api_url}/view"
        params = {"filename": filename, "subfolder": subfolder}
        
        response = requests.get(url, params=params, timeout=30)
        return response.content if response.status_code == 200 else b""
    
    def upload_image(self, image_path: str) -> Optional[str]:
        """Sube imagen para img2img"""
        url = f"{self.api_url}/upload/image"
        
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(url, files=files, timeout=60)
            
        if response.status_code == 200:
            data = response.json()
            return data.get('name')
        return None
    
    def interrupt(self):
        """Interrumpe generación actual"""
        requests.post(f"{self.api_url}/interrupt", timeout=5)
    
    def clear_queue(self):
        """Limpia cola de generación"""
        requests.post(f"{self.api_url}/queue", json={"clear": True}, timeout=5)
