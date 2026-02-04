#!/usr/bin/env python3
"""
CLIP Embedding Server for Memory Brain
Uses open-clip for cross-modal image/text embeddings on Apple Silicon.
"""

import io
import json
import base64
from pathlib import Path

from flask import Flask, request, jsonify
from PIL import Image
import open_clip
import torch

app = Flask(__name__)

# Global model
model = None
preprocess = None
tokenizer = None
device = "mps" if torch.backends.mps.is_available() else "cpu"

def load_model():
    """Load CLIP model (ViT-B-32)"""
    global model, preprocess, tokenizer
    
    print(f"🔄 Loading CLIP model on {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32',
        pretrained='laion2b_s34b_b79k'
    )
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model = model.to(device)
    model.eval()
    print("✅ CLIP model loaded!")

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "model": "ViT-B-32",
        "device": device,
        "embedding_dim": 512
    })

@app.route('/embed/image', methods=['POST'])
def embed_image():
    """Generate embedding for an image
    
    Accepts:
    - multipart/form-data with 'file' field
    - JSON with 'path' (local file path)
    - JSON with 'base64' (base64-encoded image)
    """
    try:
        if 'file' in request.files:
            # Multipart upload
            file = request.files['file']
            image = Image.open(file.stream).convert('RGB')
        elif request.is_json:
            data = request.get_json()
            if 'path' in data:
                # Local file path
                image = Image.open(data['path']).convert('RGB')
            elif 'base64' in data:
                # Base64 encoded
                img_bytes = base64.b64decode(data['base64'])
                image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            else:
                return jsonify({"error": "No image provided"}), 400
        else:
            return jsonify({"error": "Invalid request format"}), 400
        
        # Preprocess and encode
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embedding = model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # L2 normalize
        
        return jsonify({
            "embedding": embedding[0].cpu().tolist(),
            "dim": 512
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/embed/text', methods=['POST'])
def embed_text():
    """Generate embedding for text"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data['text']
        tokens = tokenizer([text]).to(device)
        
        with torch.no_grad():
            embedding = model.encode_text(tokens)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # L2 normalize
        
        return jsonify({
            "embedding": embedding[0].cpu().tolist(),
            "dim": 512
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/embed/batch', methods=['POST'])
def embed_batch():
    """Batch embed multiple images and/or texts"""
    try:
        data = request.get_json()
        results = []
        
        # Process images
        if 'images' in data:
            for img_path in data['images']:
                try:
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = preprocess(image).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        emb = model.encode_image(image_tensor)
                        emb = emb / emb.norm(dim=-1, keepdim=True)
                    
                    results.append({
                        "type": "image",
                        "path": img_path,
                        "embedding": emb[0].cpu().tolist()
                    })
                except Exception as e:
                    results.append({
                        "type": "image",
                        "path": img_path,
                        "error": str(e)
                    })
        
        # Process texts
        if 'texts' in data:
            for text in data['texts']:
                try:
                    tokens = tokenizer([text]).to(device)
                    
                    with torch.no_grad():
                        emb = model.encode_text(tokens)
                        emb = emb / emb.norm(dim=-1, keepdim=True)
                    
                    results.append({
                        "type": "text",
                        "text": text,
                        "embedding": emb[0].cpu().tolist()
                    })
                except Exception as e:
                    results.append({
                        "type": "text",
                        "text": text,
                        "error": str(e)
                    })
        
        return jsonify({"results": results})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/similarity', methods=['POST'])
def similarity():
    """Calculate similarity between image and text"""
    try:
        data = request.get_json()
        
        if 'image_path' not in data or 'text' not in data:
            return jsonify({"error": "Need both image_path and text"}), 400
        
        # Encode image
        image = Image.open(data['image_path']).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        # Encode text
        tokens = tokenizer([data['text']]).to(device)
        
        with torch.no_grad():
            image_emb = model.encode_image(image_tensor)
            text_emb = model.encode_text(tokens)
            
            # Normalize
            image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            
            # Cosine similarity
            similarity = (image_emb @ text_emb.T).item()
        
        return jsonify({
            "similarity": similarity,
            "image": data['image_path'],
            "text": data['text']
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    import sys
    
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5050
    
    load_model()
    
    print(f"\n🖼️ CLIP Server running on http://localhost:{port}")
    print("Endpoints:")
    print("  POST /embed/image  - Get image embedding")
    print("  POST /embed/text   - Get text embedding")
    print("  POST /embed/batch  - Batch embeddings")
    print("  POST /similarity   - Image-text similarity")
    print("  GET  /health       - Health check")
    
    app.run(host='0.0.0.0', port=port, debug=False)
