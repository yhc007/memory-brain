#!/usr/bin/env python3
"""
Photos Library → Memory Brain Sync

Extracts photo metadata from Apple Photos and generates CLIP embeddings.
Original photos stay in Photos app - only metadata + embeddings are stored.
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path

import osxphotos
import requests

CLIP_SERVER = os.environ.get("CLIP_SERVER_URL", "http://localhost:5050")
VLM_SERVER = os.environ.get("VLM_SERVER_URL", "http://localhost:5051")
OUTPUT_DIR = Path(__file__).parent / "visual_index"

def check_clip_server():
    """Check if CLIP server is running"""
    try:
        resp = requests.get(f"{CLIP_SERVER}/health", timeout=5)
        return resp.status_code == 200
    except:
        return False

def check_vlm_server():
    """Check if VLM server is running"""
    try:
        resp = requests.get(f"{VLM_SERVER}/health", timeout=5)
        return resp.status_code == 200
    except:
        return False

def get_caption(image_path: str, prompt: str = "Describe this photo briefly in one sentence.") -> str | None:
    """Get VLM caption for an image"""
    try:
        resp = requests.post(
            f"{VLM_SERVER}/caption",
            json={"path": image_path, "prompt": prompt, "max_tokens": 80},
            timeout=60
        )
        if resp.status_code == 200:
            caption = resp.json().get("caption", "").strip()
            # Clean up repetitive text
            sentences = caption.split('. ')
            unique = []
            for s in sentences[:3]:  # Keep first 3 sentences max
                if s and s not in unique:
                    unique.append(s)
            return '. '.join(unique)
    except Exception as e:
        print(f"  ⚠️ Caption error: {e}")
    return None

def get_embedding(image_path: str) -> list[float] | None:
    """Get CLIP embedding for an image"""
    try:
        resp = requests.post(
            f"{CLIP_SERVER}/embed/image",
            json={"path": image_path},
            timeout=30
        )
        if resp.status_code == 200:
            return resp.json().get("embedding")
    except Exception as e:
        print(f"  ⚠️ Embedding error: {e}")
    return None

def sync_photos(limit: int = None, album: str = None, verbose: bool = True):
    """
    Sync photos from Apple Photos to Memory Brain
    
    Args:
        limit: Max number of photos to sync (None = all)
        album: Specific album name to sync (None = all)
        verbose: Print progress
    """
    if not check_clip_server():
        print("❌ CLIP server not available")
        print(f"   Start it: python clip_server.py 5050")
        return
    
    print(f"📸 Opening Photos Library...")
    photosdb = osxphotos.PhotosDB()
    
    # Get photos
    if album:
        albums = [a for a in photosdb.album_info if a.title == album]
        if not albums:
            print(f"❌ Album '{album}' not found")
            print(f"   Available albums: {[a.title for a in photosdb.album_info]}")
            return
        photos = albums[0].photos
        print(f"📁 Album: {album} ({len(photos)} photos)")
    else:
        photos = photosdb.photos()
        print(f"📚 Total photos: {len(photos)}")
    
    if limit:
        photos = photos[:limit]
        print(f"   Processing first {limit} photos")
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Process each photo
    results = []
    for i, photo in enumerate(photos):
        if verbose:
            print(f"\n[{i+1}/{len(photos)}] {photo.original_filename}")
        
        # Get the actual file path (try original, then derivatives)
        path = photo.path
        if not path or not os.path.exists(path):
            # Try derivatives (thumbnails/previews for iCloud-optimized photos)
            if photo.path_derivatives:
                # Prefer larger derivative (masters)
                for deriv in photo.path_derivatives:
                    if os.path.exists(deriv):
                        path = deriv
                        break
        
        if not path or not os.path.exists(path):
            if verbose:
                print(f"  ⚠️ File not accessible (iCloud only?)")
            continue
        
        # Get CLIP embedding
        if verbose:
            print(f"  🔄 Generating embedding...")
        embedding = get_embedding(path)
        
        if not embedding:
            if verbose:
                print(f"  ⚠️ Failed to get embedding")
            continue
        
        # Get VLM caption (if server available)
        caption = None
        if check_vlm_server():
            if verbose:
                print(f"  🧠 Generating caption...")
            caption = get_caption(path)
            if caption and verbose:
                print(f"     📝 {caption[:60]}...")
        
        # Extract metadata
        metadata = {
            "uuid": photo.uuid,
            "filename": photo.original_filename,
            "path": path,
            "date": photo.date.isoformat() if photo.date else None,
            "description": photo.description or "",
            "title": photo.title or "",
            "keywords": [str(k) for k in photo.keywords] if photo.keywords else [],
            "albums": [str(a.title) for a in photo.albums] if photo.albums else [],
            "persons": [str(p.name) if hasattr(p, 'name') else str(p) for p in photo.persons] if photo.persons else [],
            "place": photo.place.name if photo.place else None,
            "location": {
                "lat": photo.latitude,
                "lon": photo.longitude
            } if photo.latitude else None,
            "is_favorite": photo.favorite,
            "is_screenshot": photo.screenshot,
            "is_selfie": photo.selfie,
            "width": photo.width,
            "height": photo.height,
            "caption": caption,
            "embedding": embedding,
            "synced_at": datetime.now().isoformat()
        }
        
        results.append(metadata)
        
        if verbose:
            tags = [str(t) for t in (metadata["keywords"] + metadata["albums"] + metadata["persons"])]
            print(f"  ✅ Synced: {metadata['filename']}")
            if tags:
                print(f"     Tags: {', '.join(tags[:5])}")
            if metadata["place"]:
                print(f"     Place: {metadata['place']}")
    
    # Save index
    index_file = OUTPUT_DIR / "photos_index.json"
    
    # Load existing index
    existing = []
    if index_file.exists():
        with open(index_file) as f:
            existing = json.load(f)
    
    # Merge (update existing, add new)
    existing_uuids = {p["uuid"] for p in existing}
    for result in results:
        if result["uuid"] in existing_uuids:
            # Update existing
            for i, p in enumerate(existing):
                if p["uuid"] == result["uuid"]:
                    existing[i] = result
                    break
        else:
            # Add new
            existing.append(result)
    
    # Save
    with open(index_file, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Synced {len(results)} photos")
    print(f"📁 Index saved to: {index_file}")
    print(f"   Total indexed: {len(existing)} photos")
    
    return results

def list_albums():
    """List all albums in Photos Library"""
    photosdb = osxphotos.PhotosDB()
    print("📚 Albums in Photos Library:\n")
    for album in sorted(photosdb.album_info, key=lambda a: a.title):
        print(f"  📁 {album.title} ({len(album.photos)} photos)")

def search_photos(query: str, limit: int = 10):
    """Search indexed photos by text"""
    if not check_clip_server():
        print("❌ CLIP server not available")
        return
    
    index_file = OUTPUT_DIR / "photos_index.json"
    if not index_file.exists():
        print("❌ No photos indexed yet. Run: python photos_sync.py sync")
        return
    
    # Get query embedding
    resp = requests.post(f"{CLIP_SERVER}/embed/text", json={"text": query})
    if resp.status_code != 200:
        print("❌ Failed to get query embedding")
        return
    
    query_embedding = resp.json()["embedding"]
    
    # Load index
    with open(index_file) as f:
        photos = json.load(f)
    
    # Calculate similarities
    def cosine_sim(a, b):
        dot = sum(x*y for x, y in zip(a, b))
        norm_a = sum(x*x for x in a) ** 0.5
        norm_b = sum(x*x for x in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0
    
    scored = []
    for photo in photos:
        if "embedding" in photo:
            sim = cosine_sim(query_embedding, photo["embedding"])
            scored.append((photo, sim))
    
    # Sort by similarity
    scored.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🔍 Search: \"{query}\"\n")
    for photo, sim in scored[:limit]:
        print(f"  [{sim:.3f}] {photo['filename']}")
        if photo.get("description"):
            print(f"          📝 {photo['description']}")
        if photo.get("place"):
            print(f"          📍 {photo['place']}")
        if photo.get("persons"):
            print(f"          👤 {', '.join(photo['persons'][:3])}")
        print()

def stats():
    """Show stats about indexed photos"""
    index_file = OUTPUT_DIR / "photos_index.json"
    if not index_file.exists():
        print("❌ No photos indexed yet")
        return
    
    with open(index_file) as f:
        photos = json.load(f)
    
    # Collect stats
    total = len(photos)
    with_location = sum(1 for p in photos if p.get("location"))
    with_persons = sum(1 for p in photos if p.get("persons"))
    favorites = sum(1 for p in photos if p.get("is_favorite"))
    
    all_persons = set()
    all_keywords = set()
    all_albums = set()
    for p in photos:
        all_persons.update(p.get("persons", []))
        all_keywords.update(p.get("keywords", []))
        all_albums.update(p.get("albums", []))
    
    print("📊 Visual Memory Index Stats\n")
    print(f"  📷 Total photos: {total}")
    print(f"  📍 With location: {with_location}")
    print(f"  👤 With faces: {with_persons}")
    print(f"  ⭐ Favorites: {favorites}")
    print(f"\n  👥 Unique persons: {len(all_persons)}")
    print(f"  🏷️ Unique keywords: {len(all_keywords)}")
    print(f"  📁 Albums: {len(all_albums)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sync Apple Photos to Memory Brain")
    subparsers = parser.add_subparsers(dest="command")
    
    # sync command
    sync_parser = subparsers.add_parser("sync", help="Sync photos from Apple Photos")
    sync_parser.add_argument("--limit", type=int, help="Max photos to sync")
    sync_parser.add_argument("--album", type=str, help="Specific album to sync")
    sync_parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")
    
    # albums command
    subparsers.add_parser("albums", help="List all albums")
    
    # search command
    search_parser = subparsers.add_parser("search", help="Search indexed photos")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", type=int, default=10, help="Max results")
    
    # stats command
    subparsers.add_parser("stats", help="Show index statistics")
    
    args = parser.parse_args()
    
    if args.command == "sync":
        sync_photos(limit=args.limit, album=args.album, verbose=not args.quiet)
    elif args.command == "albums":
        list_albums()
    elif args.command == "search":
        search_photos(args.query, args.limit)
    elif args.command == "stats":
        stats()
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python photos_sync.py albums           # List all albums")
        print("  python photos_sync.py sync --limit 10  # Sync first 10 photos")
        print("  python photos_sync.py sync --album '최근 항목'  # Sync specific album")
        print("  python photos_sync.py search 'beach sunset'    # Search by text")
        print("  python photos_sync.py stats            # Show statistics")
