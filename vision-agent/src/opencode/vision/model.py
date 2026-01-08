"""LFM2.5-VL Vision-Language Model Loader"""

import torch
from typing import Optional
from PIL import Image


class VisionModel:
    """LFM2.5-VL-1.6B wrapper for vision understanding"""

    MODEL_ID = "LiquidAI/LFM2.5-VL-1.6B"

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None

    def load(self):
        """Load model and processor"""
        from transformers import AutoProcessor, AutoModelForImageTextToText

        print(f"Loading {self.MODEL_ID} on {self.device}...")

        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.MODEL_ID,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        )

        if self.device != "cuda":
            self.model = self.model.to(self.device)

        print("Model loaded!")
        return self

    def analyze(self, image: Image.Image, prompt: str, max_tokens: int = 256) -> str:
        """Analyze image with a prompt"""
        if self.model is None:
            self.load()

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=True,
        )

        response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        # Extract assistant response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1].strip()
        return response

    def describe_screen(self, image: Image.Image) -> str:
        """Describe what's on screen"""
        return self.analyze(
            image,
            "Describe what you see on this screen. Identify UI elements, text, buttons, and their positions."
        )

    def find_element(self, image: Image.Image, element: str) -> str:
        """Find a specific UI element"""
        return self.analyze(
            image,
            f"Find the '{element}' on this screen. Describe its exact location (top/bottom, left/right, center)."
        )

    def read_error(self, image: Image.Image) -> str:
        """Read and explain error messages"""
        return self.analyze(
            image,
            "Read any error messages on this screen. Explain what the error means and suggest how to fix it."
        )

    def extract_text(self, image: Image.Image) -> str:
        """OCR - extract all visible text"""
        return self.analyze(
            image,
            "Extract and list all visible text from this image, preserving the layout as much as possible."
        )
