import os
try:
    from diffusers import StableDiffusionPipeline
    import torch
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

from PIL import Image, ImageDraw, ImageFont
import numpy as np

class ImageGenerator:
    def __init__(self):
        self.pipe = None
        self.model_id = "runwayml/stable-diffusion-v1-5"  # Standard model
        
    def load_model(self):
        if DIFFUSERS_AVAILABLE and not self.pipe:
            try:
                print("Loading Stable Diffusion...")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    self.model_id, 
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                )
                self.pipe = self.pipe.to(device)
                print(f"Model loaded on {device}")
                return True
            except Exception as e:
                print(f"Failed to load Stable Diffusion: {e}")
                return False
        return False

    def generate(self, prompt, output_path="generated_image.png"):
        """
        Generate image from prompt.
        If diffusers is not available, generate a placeholder image using Pillow.
        """
        if DIFFUSERS_AVAILABLE:
            if not self.pipe:
                success = self.load_model()
                if not success:
                    return self._generate_placeholder(prompt, output_path, "Model Load Failed")
            
            try:
                image = self.pipe(prompt).images[0]
                image.save(output_path)
                return output_path
            except Exception as e:
                print(f"Generation failed: {e}")
                return self._generate_placeholder(prompt, output_path, f"Error: {str(e)}")
        else:
            return self._generate_placeholder(prompt, output_path, "Install 'diffusers' & 'torch' for real AI images")

    def _generate_placeholder(self, prompt, output_path, message):
        """Create a placeholder image if AI generation fails/unavailable."""
        width, height = 512, 512
        # varying background based on prompt length
        color = (len(prompt) * 5 % 255, len(prompt) * 10 % 255, len(prompt) * 15 % 255)
        img = Image.new('RGB', (width, height), color=color)
        d = ImageDraw.Draw(img)
        
        # Add text
        try:
            # Try to use a default font
            font = ImageFont.load_default()
        except:
            font = None

        # Position text
        d.text((20, 200), "Image Generation Placeholder", fill=(255, 255, 255))
        d.text((20, 230), f"Prompt: {prompt[:30]}...", fill=(255, 255, 255)) 
        d.text((20, 260), message, fill=(255, 255, 0)) # Yellow warning
        
        img.save(output_path)
        return output_path
