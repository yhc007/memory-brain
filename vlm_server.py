#!/usr/bin/env python3
"""
VLM Server for Memory Brain
Uses mlx-vlm for image captioning on Apple Silicon.
"""

import os
import sys
from pathlib import Path

# Set HF token from environment
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

# Global model
model = None
processor = None

def load_model():
    """Load VLM model"""
    global model, processor
    
    from mlx_vlm import load, generate
    from mlx_vlm.utils import load_config
    
    print("🔄 Loading VLM model (this may take a while)...")
    
    # Use Qwen2-VL - works great on Apple Silicon!
    model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
    
    model, processor = load(model_path)
    
    print(f"✅ VLM model loaded: {model_path}")

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        "status": "ok",
        "model": "Qwen2-VL-2B-Instruct-4bit",
        "type": "vlm"
    })

@app.route('/caption', methods=['POST'])
def caption():
    """Generate caption for an image"""
    from mlx_vlm import generate
    
    try:
        data = request.get_json()
        
        if 'path' not in data:
            return jsonify({"error": "No image path provided"}), 400
        
        image_path = data['path']
        if not os.path.exists(image_path):
            return jsonify({"error": f"Image not found: {image_path}"}), 404
        
        prompt = data.get('prompt', "Describe this image in detail.")
        max_tokens = data.get('max_tokens', 200)
        
        # Generate caption
        result = generate(
            model,
            processor,
            prompt,
            image=image_path,
            max_tokens=max_tokens,
            verbose=False
        )
        
        return jsonify({
            "caption": result.text if hasattr(result, 'text') else str(result),
            "path": image_path
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/describe', methods=['POST'])
def describe():
    """Describe an image with a specific question"""
    from mlx_vlm import generate
    
    try:
        data = request.get_json()
        
        if 'path' not in data:
            return jsonify({"error": "No image path provided"}), 400
        
        image_path = data['path']
        if not os.path.exists(image_path):
            return jsonify({"error": f"Image not found: {image_path}"}), 404
        
        question = data.get('question', "What is in this image?")
        max_tokens = data.get('max_tokens', 300)
        
        result = generate(
            model,
            processor,
            question,
            image=image_path,
            max_tokens=max_tokens,
            verbose=False
        )
        
        return jsonify({
            "answer": result.text if hasattr(result, 'text') else str(result),
            "question": question,
            "path": image_path
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/batch_caption', methods=['POST'])
def batch_caption():
    """Generate captions for multiple images"""
    from mlx_vlm import generate
    
    try:
        data = request.get_json()
        
        if 'paths' not in data:
            return jsonify({"error": "No image paths provided"}), 400
        
        paths = data['paths']
        prompt = data.get('prompt', "Describe this image briefly in one sentence.")
        max_tokens = data.get('max_tokens', 100)
        
        results = []
        for path in paths:
            if not os.path.exists(path):
                results.append({"path": path, "error": "File not found"})
                continue
            
            try:
                result = generate(
                    model,
                    processor,
                    prompt,
                    image=path,
                    max_tokens=max_tokens,
                    verbose=False
                )
                caption = result.text if hasattr(result, 'text') else str(result)
                results.append({"path": path, "caption": caption})
            except Exception as e:
                results.append({"path": path, "error": str(e)})
        
        return jsonify({"results": results})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5051
    
    load_model()
    
    print(f"\n🧠 VLM Server running on http://localhost:{port}")
    print("Endpoints:")
    print("  POST /caption       - Generate image caption")
    print("  POST /describe      - Answer question about image")
    print("  POST /batch_caption - Batch captioning")
    print("  GET  /health        - Health check")
    
    app.run(host='0.0.0.0', port=port, debug=False)
