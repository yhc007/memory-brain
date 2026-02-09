//! Web UI for Memory Brain
//! 
//! Beautiful dashboard with HTMX + Lucide icons

use axum::{
    extract::State,
    response::Html,
    routing::get,
    Router,
    Form,
};
use std::sync::Arc;
use serde::Deserialize;

use crate::server::AppState;
use crate::audit;

// Load templates at compile time
const BASE_TEMPLATE: &str = include_str!("../templates/base.html");
const SEARCH_TEMPLATE: &str = include_str!("../templates/search.html");
const STORE_TEMPLATE: &str = include_str!("../templates/store.html");

/// Render page with base template
fn render_page(title: &str, content: &str) -> String {
    BASE_TEMPLATE
        .replace("{{TITLE}}", title)
        .replace("{{CONTENT}}", content)
}

/// Dashboard page
pub async fn dashboard_page(State(state): State<Arc<AppState>>) -> Html<String> {
    let brain = state.brain.read().await;
    let (stores, recalls, searches) = audit::get_daily_stats();
    let total = stores + recalls + searches;
    
    // Get memory count
    let memory_count = brain.semantic.search("", 10000).map(|v| v.len()).unwrap_or(0);
    
    let store_pct = if total > 0 { stores * 100 / total } else { 0 };
    let recall_pct = if total > 0 { recalls * 100 / total } else { 0 };
    let search_pct = if total > 0 { searches * 100 / total } else { 0 };
    
    let content = format!(
        r##"<div class="mb-6 sm:mb-8">
    <h1 class="text-xl sm:text-2xl font-semibold tracking-tight">Dashboard</h1>
    <p class="text-zinc-500 text-sm mt-1">Memory Brain overview and activity</p>
</div>

<div class="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 mb-6 sm:mb-8">
    <div class="stat-card">
        <div class="flex items-center gap-1.5 text-zinc-500 text-xs font-medium uppercase tracking-wider mb-2">
            <i data-lucide="archive" class="w-3.5 h-3.5"></i> <span class="hidden sm:inline">Stores</span><span class="sm:hidden">Store</span>
        </div>
        <div class="text-2xl sm:text-3xl font-semibold text-zinc-100">{}</div>
        <div class="text-xs text-zinc-600 mt-1">today</div>
    </div>
    <div class="stat-card">
        <div class="flex items-center gap-1.5 text-zinc-500 text-xs font-medium uppercase tracking-wider mb-2">
            <i data-lucide="search" class="w-3.5 h-3.5"></i> Recalls
        </div>
        <div class="text-2xl sm:text-3xl font-semibold text-zinc-100">{}</div>
        <div class="text-xs text-zinc-600 mt-1">today</div>
    </div>
    <div class="stat-card">
        <div class="flex items-center gap-1.5 text-zinc-500 text-xs font-medium uppercase tracking-wider mb-2">
            <i data-lucide="filter" class="w-3.5 h-3.5"></i> <span class="hidden sm:inline">Searches</span><span class="sm:hidden">Search</span>
        </div>
        <div class="text-2xl sm:text-3xl font-semibold text-zinc-100">{}</div>
        <div class="text-xs text-zinc-600 mt-1">today</div>
    </div>
    <div class="stat-card glow-subtle">
        <div class="flex items-center gap-1.5 text-indigo-400 text-xs font-medium uppercase tracking-wider mb-2">
            <i data-lucide="brain" class="w-3.5 h-3.5"></i> <span class="hidden sm:inline">Memories</span><span class="sm:hidden">Total</span>
        </div>
        <div class="text-2xl sm:text-3xl font-semibold text-zinc-100">{}</div>
        <div class="text-xs text-zinc-600 mt-1">total</div>
    </div>
</div>

<div class="grid grid-cols-1 md:grid-cols-2 gap-3 sm:gap-4">
    <div class="card p-4 sm:p-6">
        <h2 class="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-4">Activity</h2>
        <div class="space-y-3">
            <div class="flex items-center gap-3">
                <span class="text-zinc-400 text-sm w-16 sm:w-20">Stores</span>
                <div class="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                    <div class="h-full bg-emerald-500 rounded-full transition-all" style="width: {}%"></div>
                </div>
                <span class="text-zinc-500 text-xs mono w-8 text-right">{}</span>
            </div>
            <div class="flex items-center gap-3">
                <span class="text-zinc-400 text-sm w-16 sm:w-20">Recalls</span>
                <div class="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                    <div class="h-full bg-blue-500 rounded-full transition-all" style="width: {}%"></div>
                </div>
                <span class="text-zinc-500 text-xs mono w-8 text-right">{}</span>
            </div>
            <div class="flex items-center gap-3">
                <span class="text-zinc-400 text-sm w-16 sm:w-20">Searches</span>
                <div class="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                    <div class="h-full bg-amber-500 rounded-full transition-all" style="width: {}%"></div>
                </div>
                <span class="text-zinc-500 text-xs mono w-8 text-right">{}</span>
            </div>
        </div>
    </div>
    
    <div class="card p-4 sm:p-6">
        <h2 class="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-4">Quick Actions</h2>
        <div class="flex flex-wrap gap-2">
            <a href="/store" class="flex items-center gap-2 bg-indigo-500/10 text-indigo-400 hover:bg-indigo-500/20 px-4 py-2.5 rounded-lg transition text-sm font-medium">
                <i data-lucide="plus" class="w-3.5 h-3.5"></i> Store
            </a>
            <a href="/search" class="flex items-center gap-2 bg-zinc-800 text-zinc-300 hover:bg-zinc-700 px-4 py-2.5 rounded-lg transition text-sm font-medium">
                <i data-lucide="search" class="w-3.5 h-3.5"></i> Search
            </a>
            <a href="/memories" class="flex items-center gap-2 bg-zinc-800 text-zinc-300 hover:bg-zinc-700 px-4 py-2.5 rounded-lg transition text-sm font-medium">
                <i data-lucide="list" class="w-3.5 h-3.5"></i> Browse
            </a>
        </div>
    </div>
</div>"##,
        stores, recalls, searches, memory_count,
        store_pct, stores, recall_pct, recalls, search_pct, searches
    );
    
    Html(render_page("Dashboard", &content))
}

/// Memories list page
pub async fn memories_page(State(state): State<Arc<AppState>>) -> Html<String> {
    let brain = state.brain.read().await;
    let memories = brain.semantic.search("", 50).unwrap_or_default();
    
    let mut memory_cards = String::new();
    for mem in memories {
        let tags_html: String = mem.tags.iter()
            .map(|t| format!(r#"<span class="badge bg-indigo-500/10 text-indigo-400">#{}</span>"#, t))
            .collect::<Vec<_>>()
            .join(" ");
        
        memory_cards.push_str(&format!(
            r##"<div class="card p-5">
                <p class="text-zinc-300 text-sm leading-relaxed mb-3">{}</p>
                <div class="flex justify-between items-center">
                    <div class="flex gap-1.5">{}</div>
                    <span class="text-zinc-600 text-xs mono">{}</span>
                </div>
            </div>"##, 
            html_escape(&mem.content),
            tags_html,
            &mem.id.to_string()[..8]
        ));
    }
    
    let content = format!(
        r##"<div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6 sm:mb-8">
            <div>
                <h1 class="text-2xl font-semibold tracking-tight">Memories</h1>
                <p class="text-zinc-500 text-sm mt-1">All stored memories</p>
            </div>
            <a href="/store" class="flex items-center gap-2 bg-indigo-500/15 text-indigo-400 hover:bg-indigo-500/25 px-4 py-2.5 rounded-lg transition text-sm font-medium">
                <i data-lucide="plus" class="w-3.5 h-3.5"></i> Store New
            </a>
        </div>
        <div class="space-y-3">{}</div>"##,
        if memory_cards.is_empty() { 
            r#"<div class="text-center text-zinc-600 py-16">No memories yet. <a href="/store" class="text-indigo-400 hover:text-indigo-300 transition">Store your first memory</a></div>"#.to_string()
        } else { 
            memory_cards 
        }
    );
    
    Html(render_page("Memories", &content))
}

/// Search page
pub async fn search_page() -> Html<String> {
    Html(render_page("Search", SEARCH_TEMPLATE))
}

/// Search results (HTMX partial)
#[derive(Deserialize)]
#[allow(dead_code)]
pub struct SearchForm {
    query: String,
    #[serde(default)]
    tags: String,
    #[serde(default)]
    from_date: String,
    #[serde(default)]
    to_date: String,
    #[serde(default)]
    limit: Option<usize>,
}

pub async fn search_results(
    State(state): State<Arc<AppState>>,
    Form(form): Form<SearchForm>,
) -> Html<String> {
    if form.query.is_empty() && form.tags.is_empty() {
        return Html(r#"<div class="text-zinc-600 text-center py-10 text-sm">Enter a search query or select tags</div>"#.to_string());
    }
    
    let limit = form.limit.unwrap_or(20);
    let mut brain = state.brain.write().await;
    
    // 쿼리가 있으면 recall, 없으면 전체에서 필터
    let memories = if !form.query.is_empty() {
        brain.recall(&form.query, limit)
    } else {
        brain.semantic.search("", limit).unwrap_or_default()
    };
    
    // 태그 필터 적용
    let tag_filters: Vec<&str> = form.tags
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    
    let filtered: Vec<_> = memories.into_iter()
        .filter(|mem| {
            if tag_filters.is_empty() {
                true
            } else {
                tag_filters.iter().any(|tag| 
                    mem.tags.iter().any(|t| t.to_lowercase().contains(&tag.to_lowercase()))
                )
            }
        })
        .take(limit)
        .collect();
    
    if filtered.is_empty() {
        let filter_info = if !form.tags.is_empty() {
            format!(" with tags '{}'", form.tags)
        } else {
            String::new()
        };
        return Html(format!(
            r##"<div class="text-center py-14">
                <i data-lucide="search-x" class="w-10 h-10 mx-auto text-zinc-700 mb-3"></i>
                <div class="text-zinc-500 text-sm">No memories found for "{}"{}</div>
            </div>"##, 
            html_escape(&form.query),
            filter_info
        ));
    }
    
    let mut html = format!(
        r#"<div class="text-zinc-500 text-xs font-medium uppercase tracking-wider mb-4">Found {} memories</div>"#,
        filtered.len()
    );
    
    for mem in filtered {
        let tags_html: String = mem.tags.iter()
            .map(|t| format!(r#"<span class="badge bg-indigo-500/10 text-indigo-400">#{}</span>"#, t))
            .collect::<Vec<_>>()
            .join(" ");
        
        html.push_str(&format!(
            r##"<div class="card p-5">
                <p class="text-zinc-300 text-sm leading-relaxed mb-3">{}</p>
                <div class="flex justify-between items-center">
                    <div class="flex gap-1.5">{}</div>
                    <span class="text-zinc-600 text-xs mono">{}</span>
                </div>
            </div>"##, 
            html_escape(&mem.content),
            tags_html,
            &mem.id.to_string()[..8]
        ));
    }
    
    Html(html)
}

/// Store page
pub async fn store_page() -> Html<String> {
    Html(render_page("Store", STORE_TEMPLATE))
}

/// Store submit (HTMX partial)
#[derive(Deserialize)]
pub struct StoreForm {
    content: String,
    tags: String,
}

pub async fn store_submit(
    State(state): State<Arc<AppState>>,
    Form(form): Form<StoreForm>,
) -> Html<String> {
    if form.content.is_empty() {
        return Html(r#"<div class="text-red-400 p-4 rounded-lg bg-red-500/10 border border-red-500/20 text-sm"><i data-lucide="x-circle" class="w-4 h-4 inline-block align-middle"></i> Content is required</div>"#.to_string());
    }
    
    let mut brain = state.brain.write().await;
    let tags: Vec<String> = form.tags
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    
    let embedding = state.embedder.embed(&form.content);
    
    let mut item = crate::MemoryItem::new(&form.content, None);
    item.tags = tags;
    item.embedding = Some(embedding.clone());
    
    match brain.semantic.store(item.clone()) {
        Ok(_) => {
            let _ = state.hnsw.add(item.id, embedding);
            crate::audit::log_store(&form.content, &item.tags);
            Html(format!(
                r##"<div class="text-emerald-400 p-4 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-sm">
                    <i data-lucide="check-circle" class="w-4 h-4 inline-block align-middle"></i> Memory stored! ID: <span class="font-mono">{}</span>
                </div>
                <script>setTimeout(() => document.querySelector('form').reset(), 100);</script>"##,
                &item.id.to_string()[..8]
            ))
        }
        Err(e) => {
            Html(format!(
                r##"<div class="text-red-400 p-4 rounded-lg bg-red-500/10 border border-red-500/20 text-sm"><i data-lucide="x-circle" class="w-4 h-4 inline-block align-middle"></i> Error: {}</div>"##,
                e
            ))
        }
    }
}

// Load visual template
const VISUAL_TEMPLATE: &str = include_str!("../templates/visual.html");

/// Visual Memory page
pub async fn visual_page() -> Html<String> {
    // Try to connect to CLIP server
    let clip_status = match ureq::get("http://localhost:5050/health").call() {
        Ok(resp) => {
            if resp.status() == 200 {
                "<span class=\"w-2 h-2 rounded-full bg-emerald-500 inline-block mr-1\"></span> Connected"
            } else {
                "<span class=\"w-2 h-2 rounded-full bg-yellow-500 inline-block\"></span> Error"
            }
        }
        Err(_) => "<span class=\"w-2 h-2 rounded-full bg-red-500 inline-block\"></span> Offline"
    };
    
    // Load photos index to get count and recent photos
    let index_path = std::path::Path::new("visual_index/photos_index.json");
    let (total_images, gallery_html) = if index_path.exists() {
        let content = std::fs::read_to_string(index_path).unwrap_or_default();
        let photos: Vec<serde_json::Value> = serde_json::from_str(&content).unwrap_or_default();
        let count = photos.len();
        
        // Show recent 8 photos
        let mut gallery = String::new();
        for photo in photos.iter().take(8) {
            let filename = photo.get("filename").and_then(|f| f.as_str()).unwrap_or("?");
            let path = photo.get("path").and_then(|p| p.as_str()).unwrap_or("");
            let place = photo.get("place").and_then(|p| p.as_str()).unwrap_or("");
            
            gallery.push_str(&format!(
                r##"<div class="card p-2">
                    <div class="aspect-square bg-gray-700 rounded-lg mb-2 flex items-center justify-center overflow-hidden">
                        <img src="/api/visual/thumb?path={}" alt="{}" class="w-full h-full object-cover" onerror="this.style.display='none'" />
                    </div>
                    <div class="text-xs text-gray-400 truncate">{}</div>
                    {}
                </div>"##,
                urlencoding::encode(path),
                html_escape(filename),
                html_escape(filename),
                if !place.is_empty() { format!(r#"<div class="text-xs text-zinc-600 truncate flex items-center gap-1"><i data-lucide="map-pin" class="w-3 h-3"></i>{}</div>"#, html_escape(place)) } else { String::new() }
            ));
        }
        
        if gallery.is_empty() {
            (count.to_string(), r#"<div class="col-span-4 text-center text-zinc-600 py-14 text-sm">No photos with thumbnails available</div>"#.to_string())
        } else {
            (count.to_string(), gallery)
        }
    } else {
        ("0".to_string(), r#"<div class="col-span-4 text-center text-zinc-600 py-14">
            <div class="text-4xl mb-4"><i data-lucide="image-off" class="w-12 h-12 mx-auto text-gray-400"></i></div>
            <div>No photos indexed yet</div>
            <div class="text-sm mt-2">Run: <code class="bg-gray-700 px-2 py-1 rounded">python photos_sync.py sync --limit 100</code></div>
        </div>"#.to_string())
    };
    
    let content = VISUAL_TEMPLATE
        .replace("{{TOTAL_IMAGES}}", &total_images)
        .replace("{{CLIP_STATUS}}", clip_status)
        .replace("{{GALLERY}}", &gallery_html);
    
    Html(render_page("Visual Memory", &content))
}

/// Visual search API (HTMX)
#[derive(Deserialize)]
pub struct VisualSearchForm {
    query: String,
}

pub async fn visual_search(Form(form): Form<VisualSearchForm>) -> Html<String> {
    if form.query.is_empty() {
        return Html(r#"<div class="text-gray-500 text-center py-8">Enter a text query to search images</div>"#.to_string());
    }
    
    // Load photos index
    let index_path = std::path::Path::new("visual_index/photos_index.json");
    let photos: Vec<serde_json::Value> = if index_path.exists() {
        let content = std::fs::read_to_string(index_path).unwrap_or_default();
        serde_json::from_str(&content).unwrap_or_default()
    } else {
        return Html(r#"<div class="text-amber-400/80 text-center py-10 text-sm">No photos indexed yet. Run: <code class="bg-gray-700 px-2 py-1 rounded">python photos_sync.py sync --limit 100</code></div>"#.to_string());
    };
    
    // Get query embedding from CLIP server
    let body = serde_json::json!({ "text": form.query });
    
    let query_embedding: Vec<f64> = match ureq::post("http://localhost:5050/embed/text").send_json(body) {
        Ok(resp) => {
            let result: serde_json::Value = resp.into_json().unwrap_or_default();
            result.get("embedding")
                .and_then(|e| e.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect())
                .unwrap_or_default()
        }
        Err(_e) => {
            return Html(format!(
                r##"<div class="bg-amber-500/10 border border-amber-500/20 rounded-lg p-4">
                    <div class="text-amber-400 text-sm"><i data-lucide="alert-triangle" class="w-4 h-4 inline-block align-middle"></i> CLIP server not available</div>
                    <div class="text-zinc-500 text-xs mt-2">Start it with: <code class="bg-gray-700 px-2 py-1 rounded">python clip_server.py 5050</code></div>
                </div>"##
            ));
        }
    };
    
    if query_embedding.is_empty() {
        return Html(r#"<div class="text-red-400">Error generating embedding</div>"#.to_string());
    }
    
    // Calculate similarities
    let mut scored: Vec<(f64, &serde_json::Value)> = photos.iter()
        .filter_map(|photo| {
            let emb: Vec<f64> = photo.get("embedding")
                .and_then(|e| e.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect())?;
            
            let sim = cosine_similarity(&query_embedding, &emb);
            Some((sim, photo))
        })
        .collect();
    
    // Sort by similarity (descending)
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    
    // Build results HTML
    let mut html = format!(
        r#"<div class="text-gray-400 text-sm mb-4"><i data-lucide="search" class="w-4 h-4 inline-block align-middle"></i> Found {} photos for "{}"</div>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">"#,
        scored.len().min(12),
        html_escape(&form.query)
    );
    
    for (sim, photo) in scored.iter().take(12) {
        let filename = photo.get("filename").and_then(|f| f.as_str()).unwrap_or("?");
        let place = photo.get("place").and_then(|p| p.as_str()).unwrap_or("");
        let persons: Vec<&str> = photo.get("persons")
            .and_then(|p| p.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();
        let path = photo.get("path").and_then(|p| p.as_str()).unwrap_or("");
        
        html.push_str(&format!(
            r##"<div class="card p-3">
                <div class="aspect-square bg-gray-700 rounded-lg mb-2 flex items-center justify-center overflow-hidden">
                    <img src="/api/visual/thumb?path={}" alt="{}" class="w-full h-full object-cover" onerror="this.style.display='none'" />
                </div>
                <div class="text-sm text-gray-300 truncate">{}</div>
                <div class="text-xs text-indigo-400 mono">{:.1}%</div>
                {}
                {}
            </div>"##,
            urlencoding::encode(path),
            html_escape(filename),
            html_escape(filename),
            sim * 100.0,
            if !place.is_empty() { format!(r#"<div class="text-xs text-zinc-600 truncate flex items-center gap-1"><i data-lucide="map-pin" class="w-3 h-3"></i>{}</div>"#, html_escape(place)) } else { String::new() },
            if !persons.is_empty() { format!(r#"<div class="text-xs text-zinc-600 truncate flex items-center gap-1"><i data-lucide="user" class="w-3 h-3"></i>{}</div>"#, persons.join(", ")) } else { String::new() }
        ));
    }
    
    html.push_str("</div>");
    Html(html)
}

/// Cosine similarity between two vectors
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot / (norm_a * norm_b) }
}

/// Visual store API (HTMX)
#[derive(Deserialize)]
pub struct VisualStoreForm {
    path: String,
    description: String,
    #[serde(default)]
    tags: String,
}

pub async fn visual_store(Form(form): Form<VisualStoreForm>) -> Html<String> {
    if form.path.is_empty() {
        return Html(r#"<div class="text-red-400">Image path is required</div>"#.to_string());
    }
    
    // Check if file exists
    if !std::path::Path::new(&form.path).exists() {
        return Html(format!(
            r#"<div class="text-red-400">File not found: {}</div>"#,
            html_escape(&form.path)
        ));
    }
    
    // Try to get image embedding from CLIP server
    let body = serde_json::json!({ "path": form.path });
    
    match ureq::post("http://localhost:5050/embed/image").send_json(body) {
        Ok(resp) => {
            let result: serde_json::Value = resp.into_json().unwrap_or_default();
            if result.get("embedding").is_some() {
                let tags: Vec<&str> = form.tags.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
                Html(format!(
                    r##"<div class="bg-emerald-500/10 border border-emerald-500/20 rounded-lg p-4">
                        <div class="text-green-400"><i data-lucide="check-circle" class="w-4 h-4 inline-block align-middle"></i> Visual memory stored!</div>
                        <div class="text-zinc-500 text-xs mt-2">Path: {}</div>
                        <div class="text-zinc-500 text-xs">Description: {}</div>
                        {}
                        <div class="text-zinc-600 text-xs mt-2">(Visual storage persistence coming soon!)</div>
                    </div>"##,
                    html_escape(&form.path),
                    html_escape(&form.description),
                    if tags.is_empty() { String::new() } else {
                        format!(r#"<div class="text-zinc-500 text-xs">Tags: {}</div>"#, tags.join(", "))
                    }
                ))
            } else {
                Html(r#"<div class="text-red-400">Error generating image embedding</div>"#.to_string())
            }
        }
        Err(e) => {
            Html(format!(
                r##"<div class="bg-amber-500/10 border border-amber-500/20 rounded-lg p-4">
                    <div class="text-amber-400 text-sm"><i data-lucide="alert-triangle" class="w-4 h-4 inline-block align-middle"></i> CLIP server not available</div>
                    <div class="text-zinc-500 text-xs mt-2">Error: {}</div>
                </div>"##,
                e
            ))
        }
    }
}

/// Thumbnail API - serve image files
#[derive(Deserialize)]
pub struct ThumbQuery {
    path: String,
}

pub async fn visual_thumb(
    axum::extract::Query(query): axum::extract::Query<ThumbQuery>
) -> axum::response::Response<axum::body::Body> {
    use axum::http::{header, Response, StatusCode};
    use axum::body::Body;
    
    let path = std::path::Path::new(&query.path);
    
    if !path.exists() {
        return Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::empty())
            .unwrap();
    }
    
    // Read file
    match std::fs::read(path) {
        Ok(bytes) => {
            // Determine content type from extension
            let content_type = match path.extension().and_then(|e| e.to_str()) {
                Some("jpg") | Some("jpeg") => "image/jpeg",
                Some("png") => "image/png",
                Some("heic") => "image/heic",
                Some("gif") => "image/gif",
                Some("webp") => "image/webp",
                _ => "image/jpeg",
            };
            
            Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, content_type)
                .body(Body::from(bytes))
                .unwrap()
        }
        Err(_) => {
            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::empty())
                .unwrap()
        }
    }
}

/// Mind Map page - interactive D3.js force graph
pub async fn mindmap_page(State(state): State<Arc<AppState>>) -> Html<String> {
    let brain = state.brain.read().await;
    let memory_count = brain.semantic.search("", 10000).map(|v| v.len()).unwrap_or(0);
    
    let content = format!(r##"
<div class="mb-6"><h1 class="text-2xl font-semibold tracking-tight">Mind Map</h1>
<p class="text-zinc-500 text-sm mt-1">기억들의 연결을 시각적으로 탐색하세요. 노드를 드래그하거나 줌/패닝할 수 있습니다.</p>

<div class="flex flex-wrap gap-2 sm:gap-4 mb-4 sm:mb-6">
    <div class="card px-3 py-2">
        <span class="text-zinc-500 text-xs font-medium uppercase tracking-wider">Threshold</span>
        <select id="threshold" onchange="loadMindMap()" class="bg-[#19191c] text-zinc-300 rounded-lg px-2 py-1 ml-2 text-sm border border-[#2a2a2e]">
            <option value="0.3" selected>0.3 (많은 연결)</option>
            <option value="0.4">0.4 (보통)</option>
            <option value="0.5">0.5 (강한 연결만)</option>
            <option value="0.6">0.6 (매우 강한 연결)</option>
        </select>
    </div>
    <div class="card px-3 py-2">
        <span class="text-zinc-500 text-xs font-medium uppercase tracking-wider">Limit</span>
        <select id="limit" onchange="loadMindMap()" class="bg-[#19191c] text-zinc-300 rounded-lg px-2 py-1 ml-2 text-sm border border-[#2a2a2e]">
            <option value="50">50</option>
            <option value="100" selected>100</option>
            <option value="200">200</option>
            <option value="500">500</option>
        </select>
    </div>
    <div class="card px-3 py-2 text-indigo-400 text-sm">
        <i data-lucide="brain" class="w-4 h-4 inline-block align-middle"></i> {memory_count} memories
    </div>
</div>

<div id="mindmap-container" class="card relative" style="height: 70vh; overflow: hidden;">
    <div id="mindmap-loading" class="absolute inset-0 flex items-center justify-center text-gray-400">
        로딩 중...
    </div>
    <div id="tooltip" class="absolute hidden bg-[#111113] text-zinc-200 p-3 rounded-lg text-xs max-w-xs border border-[#2a2a2e] pointer-events-none z-10 shadow-xl"></div>
    <div id="legend" class="absolute top-4 right-4 bg-[#111113]/90 p-3 rounded-lg text-xs text-zinc-400 border border-[#2a2a2e]"></div>
</div>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
let simulation;
const colors = d3.schemeCategory10;

function loadMindMap() {{
    const threshold = document.getElementById('threshold').value;
    const limit = document.getElementById('limit').value;
    
    fetch(`/mindmap/data?threshold=${{threshold}}&limit=${{limit}}`)
        .then(r => r.json())
        .then(data => renderMindMap(data))
        .catch(e => {{
            document.getElementById('mindmap-loading').textContent = 'Error: ' + e;
        }});
}}

function renderMindMap(data) {{
    const container = document.getElementById('mindmap-container');
    const loading = document.getElementById('mindmap-loading');
    loading.style.display = 'none';
    
    // Clear previous
    d3.select('#mindmap-container svg').remove();
    if (simulation) simulation.stop();
    
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    if (data.nodes.length === 0) {{
        loading.style.display = 'flex';
        loading.textContent = '기억이 없습니다.';
        return;
    }}
    
    const svg = d3.select('#mindmap-container')
        .append('svg')
        .attr('width', width)
        .attr('height', height);
    
    const g = svg.append('g');
    
    svg.call(d3.zoom()
        .scaleExtent([0.1, 4])
        .on('zoom', (event) => g.attr('transform', event.transform)));
    
    simulation = d3.forceSimulation(data.nodes)
        .force('link', d3.forceLink(data.links).id(d => d.id).distance(80))
        .force('charge', d3.forceManyBody().strength(-150))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(d => d.size + 5));
    
    const link = g.append('g')
        .selectAll('line')
        .data(data.links)
        .join('line')
        .attr('stroke', 'rgba(255,255,255,0.08)')
        .attr('stroke-width', d => d.weight * 3);
    
    const node = g.append('g')
        .selectAll('g')
        .data(data.nodes)
        .join('g')
        .call(d3.drag()
            .on('start', (e) => {{ if (!e.active) simulation.alphaTarget(0.3).restart(); e.subject.fx = e.subject.x; e.subject.fy = e.subject.y; }})
            .on('drag', (e) => {{ e.subject.fx = e.x; e.subject.fy = e.y; }})
            .on('end', (e) => {{ if (!e.active) simulation.alphaTarget(0); e.subject.fx = null; e.subject.fy = null; }}));
    
    node.append('circle')
        .attr('r', d => d.size)
        .attr('fill', d => colors[d.group % 10])
        .attr('stroke', '#fff')
        .attr('stroke-width', 1.5)
        .style('cursor', 'pointer');
    
    node.append('text')
        .attr('dx', d => d.size + 4)
        .attr('dy', 4)
        .text(d => d.label)
        .attr('fill', '#a1a1aa')
        .attr('font-size', '10px')
        .style('pointer-events', 'none');
    
    // Tooltip
    const tooltip = document.getElementById('tooltip');
    node.on('mouseover', (event, d) => {{
        tooltip.classList.remove('hidden');
        tooltip.innerHTML = `<strong>${{d.label}}</strong><br><span class="text-gray-300">${{d.content}}</span>` +
            (d.tags.length ? `<div class="mt-2 text-gray-500">${{d.tags.map(t => '#'+t).join(' ')}}</div>` : '');
        tooltip.style.left = (event.offsetX + 15) + 'px';
        tooltip.style.top = (event.offsetY - 10) + 'px';
    }}).on('mouseout', () => tooltip.classList.add('hidden'));
    
    simulation.on('tick', () => {{
        link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
        node.attr('transform', d => `translate(${{d.x}},${{d.y}})`);
    }});
    
    // Legend
    const groups = [...new Set(data.nodes.map(n => n.group))];
    const tagNames = {{}};
    data.nodes.forEach(n => {{ if (n.tags.length && !tagNames[n.group]) tagNames[n.group] = n.tags[0]; }});
    const legend = document.getElementById('legend');
    legend.innerHTML = groups.slice(0, 10).map(g =>
        `<div class="flex items-center gap-2 my-1"><div class="w-3 h-3 rounded-full" style="background:${{colors[g % 10]}}"></div>${{tagNames[g] || 'Group '+g}}</div>`
    ).join('');
}}

loadMindMap();
</script>"##, memory_count = memory_count);

    Html(render_page("Mind Map", &content))
}

/// Mind Map JSON data endpoint
pub async fn mindmap_data(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> axum::Json<serde_json::Value> {
    let brain = state.brain.read().await;
    let threshold: f32 = params.get("threshold").and_then(|s| s.parse().ok()).unwrap_or(0.3);
    let limit: usize = params.get("limit").and_then(|s| s.parse().ok()).unwrap_or(100);
    
    let map = crate::mindmap::MindMap::from_brain(&brain, limit, threshold);
    
    let nodes: Vec<serde_json::Value> = map.nodes.iter().map(|n| {
        serde_json::json!({
            "id": n.id,
            "label": n.label,
            "content": n.content,
            "group": n.group,
            "size": n.size,
            "tags": n.tags,
        })
    }).collect();
    
    let links: Vec<serde_json::Value> = map.edges.iter().map(|e| {
        serde_json::json!({
            "source": e.source,
            "target": e.target,
            "weight": e.weight,
        })
    }).collect();
    
    axum::Json(serde_json::json!({
        "nodes": nodes,
        "links": links,
    }))
}

/// Timeline page - memories over time
pub async fn timeline_page(State(_state): State<Arc<AppState>>) -> Html<String> {
    let content = r##"
<div class="mb-6"><h1 class="text-2xl font-semibold tracking-tight">Timeline</h1>
<p class="text-zinc-500 text-sm mt-1">시간 순으로 기억의 흐름을 확인하세요.</p>

<div id="timeline-container" class="space-y-1" hx-get="/timeline/data" hx-trigger="load" hx-swap="innerHTML">
    <div class="text-zinc-600 text-center py-14 text-sm">로딩 중...</div>
</div>
"##;

    Html(render_page("Timeline", content))
}

/// Timeline data (HTMX partial)
pub async fn timeline_data(State(state): State<Arc<AppState>>) -> Html<String> {
    let brain = state.brain.read().await;
    let memories = brain.semantic.search("", 10000).unwrap_or_default();
    
    if memories.is_empty() {
        return Html(r#"<div class="text-zinc-600 text-center py-14 text-sm">기억이 없습니다.</div>"#.to_string());
    }
    
    // Group by date
    let mut by_date: std::collections::BTreeMap<String, Vec<&crate::MemoryItem>> = std::collections::BTreeMap::new();
    for mem in &memories {
        let date = mem.created_at.format("%Y-%m-%d").to_string();
        by_date.entry(date).or_default().push(mem);
    }
    
    let type_colors = |t: &crate::MemoryType| match t {
        crate::MemoryType::Episodic => ("blue", "Episodic"),
        crate::MemoryType::Semantic => ("cyan", "Semantic"),
        crate::MemoryType::Procedural => ("purple", "Procedural"),
        crate::MemoryType::Working => ("green", "Working"),
    };
    
    let mut html = String::new();
    
    // Reverse chronological
    for (date, mems) in by_date.iter().rev() {
        let weekday = if let Ok(d) = chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d") {
            d.format("%A").to_string()
        } else {
            String::new()
        };
        
        html.push_str(&format!(
            r##"<div class="mb-8">
                <div class="flex items-center gap-2 sm:gap-3 mb-3 sm:mb-4">
                    <div class="w-2 h-2 rounded-full bg-indigo-500"></div>
                    <h2 class="text-base font-semibold text-zinc-200">{}</h2>
                    <span class="text-zinc-500 text-sm">{}</span>
                    <span class="text-zinc-600 text-xs">({} memories)</span>
                </div>
                <div class="ml-6 border-l border-[#2a2a2e] pl-6 space-y-3">"##,
            date, weekday, mems.len()
        ));
        
        for mem in mems {
            let (color, type_name) = type_colors(&mem.memory_type);
            let strength_pct = (mem.strength * 100.0) as u32;
            let content_preview = if mem.content.len() > 120 {
                format!("{}...", &mem.content.chars().take(120).collect::<String>())
            } else {
                mem.content.clone()
            };
            
            html.push_str(&format!(
                r##"<div class="card p-4">
                    <div class="flex items-start justify-between">
                        <div class="flex-1">
                            <div class="text-zinc-300 text-sm leading-relaxed">{content}</div>
                            <div class="flex items-center gap-3 mt-2 text-xs">
                                <span class="badge bg-{color}-500/10 text-{color}-400">{type_name}</span>
                                <span class="text-zinc-600 mono">{time}</span>
                                <span class="text-zinc-600 mono">{strength}%</span>
                                {tags}
                            </div>
                        </div>
                    </div>
                </div>"##,
                color = color,
                content = html_escape(&content_preview),
                type_name = type_name,
                time = mem.created_at.format("%H:%M"),
                strength = strength_pct,
                tags = if mem.tags.is_empty() { String::new() } else {
                    format!("<span class=\"text-gray-600\">{}</span>", 
                        mem.tags.iter().map(|t| format!("#{}", t)).collect::<Vec<_>>().join(" "))
                }
            ));
        }
        
        html.push_str("</div></div>");
    }
    
    Html(html)
}

/// CoreDB Browser page
pub async fn coredb_page(State(_state): State<Arc<AppState>>) -> Html<String> {
    let content = r##"
<div class="mb-6">
    <h1 class="text-2xl font-semibold tracking-tight">CoreDB Browser</h1>
    <p class="text-zinc-500 text-sm mt-1">Execute CQL queries against the memory store</p>
</div>

<div class="grid grid-cols-1 lg:grid-cols-3 gap-3 sm:gap-4">
    <!-- Query Panel -->
    <div class="lg:col-span-2">
        <div class="card p-6">
            <h2 class="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-4 flex items-center gap-2"><i data-lucide="terminal" class="w-5 h-5 inline-block align-middle"></i> CQL Query</h2>
            <form hx-post="/coredb/query" hx-target="#results" hx-swap="innerHTML" hx-indicator="#loading">
                <textarea name="query" rows="3" 
                    class="w-full bg-[#19191c] text-emerald-300 mono rounded-lg p-4 border border-[#2a2a2e] focus:border-indigo-500/50 focus:outline-none focus:ring-1 focus:ring-indigo-500/20 resize-y text-sm"
                    placeholder="SELECT * FROM memory_brain.semantic LIMIT 10">SELECT * FROM memory_brain.semantic LIMIT 20</textarea>
                <div class="flex gap-3 mt-3">
                    <button type="submit" class="px-5 py-2 bg-indigo-500/15 text-indigo-400 hover:bg-indigo-500/25 rounded-lg transition text-sm font-medium flex items-center gap-2">
                        <i data-lucide="play" class="w-4 h-4 inline-block align-middle"></i> Execute
                    </button>
                    <span id="loading" class="htmx-indicator text-zinc-500 py-2 text-sm">실행 중...</span>
                </div>
            </form>
            
            <div class="mt-3 text-xs text-zinc-600 flex items-center gap-2 flex-wrap">
                Quick queries:
                <button onclick="setQuery('SELECT * FROM memory_brain.semantic LIMIT 20')" class="text-indigo-400 hover:text-indigo-300 transition">Semantic (default)</button>
                <button onclick="setQuery('SELECT * FROM memory_brain.episodic LIMIT 20')" class="text-blue-400 hover:text-blue-300 transition">Episodic</button>
                <button onclick="setQuery('SELECT * FROM memory_brain.semantic LIMIT 20')" class="text-cyan-400 hover:text-cyan-300 transition">Semantic</button>
                <button onclick="setQuery('SELECT * FROM memory_brain.procedural LIMIT 20')" class="text-purple-400 hover:text-purple-300 transition">Procedural</button>
                <button onclick="setQuery('SELECT * FROM visual_brain.visual_memories LIMIT 20')" class="text-pink-400 hover:text-pink-300 transition">Visual</button>
            </div>
        </div>
        
        <!-- Results -->
        <div id="results" class="mt-6">
            <div class="text-zinc-600 text-center py-10 text-sm">쿼리를 실행하면 결과가 여기에 표시됩니다.</div>
        </div>
    </div>
    
    <!-- Info Panel -->
    <div class="space-y-4">
        <div class="card p-6">
            <h3 class="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-3 flex items-center gap-2"><i data-lucide="table-2" class="w-5 h-5 inline-block align-middle"></i> Tables</h3>
            <div class="space-y-2 text-sm">
                <div class="flex justify-between text-gray-300">
                    <span>memory_brain.episodic</span>
                    <span class="text-zinc-600">일화 기억</span>
                </div>
                <div class="flex justify-between text-gray-300">
                    <span>memory_brain.semantic</span>
                    <span class="text-zinc-600">의미 기억</span>
                </div>
                <div class="flex justify-between text-gray-300">
                    <span>memory_brain.procedural</span>
                    <span class="text-zinc-600">절차 기억</span>
                </div>
                <div class="flex justify-between text-gray-300">
                    <span>visual_brain.visual_memories</span>
                    <span class="text-zinc-600">시각 기억</span>
                </div>
            </div>
        </div>
        
        <div class="card p-6">
            <h3 class="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-3 flex items-center gap-2"><i data-lucide="book-open" class="w-5 h-5 inline-block align-middle"></i> CQL Guide</h3>
            <div class="text-xs text-zinc-500 space-y-1.5 mono">
                <p>SELECT * FROM ks.table</p>
                <p>SELECT * FROM ks.table LIMIT n</p>
                <p>SELECT * FROM ks.table WHERE id = 'xxx'</p>
                <p>INSERT INTO ks.table (...) VALUES (...)</p>
                <p>DELETE FROM ks.table WHERE id = 'xxx'</p>
            </div>
        </div>
    </div>
</div>

<script>
function setQuery(q) {
    document.querySelector('textarea[name=query]').value = q;
}
</script>
"##;

    Html(render_page("CoreDB Browser", content))
}

/// Execute CQL query (HTMX endpoint)
#[derive(Deserialize)]
pub struct CqlQueryForm {
    query: String,
}

pub async fn coredb_query(
    State(state): State<Arc<AppState>>,
    Form(form): Form<CqlQueryForm>,
) -> Html<String> {
    let brain = state.brain.read().await;
    let query = form.query.trim();
    
    if query.is_empty() {
        return Html(r#"<div class="text-red-400">쿼리를 입력하세요.</div>"#.to_string());
    }
    
    // Execute CQL through the brain's storage
    match brain.storage_execute_cql(query) {
        Ok(result) => {
            Html(format!(
                r##"<div class="card p-6">
                    <div class="flex justify-between items-center mb-4">
                        <h3 class="text-sm font-semibold text-zinc-400 uppercase tracking-wider flex items-center gap-2"><i data-lucide="table" class="w-5 h-5 inline-block align-middle"></i> Results</h3>
                        <span class="text-xs text-zinc-600 mono">{}</span>
                    </div>
                    <div class="overflow-x-auto">
                        {}
                    </div>
                </div>"##,
                html_escape(query),
                result
            ))
        }
        Err(e) => {
            Html(format!(
                r##"<div class="bg-red-500/10 rounded-lg p-4 border border-red-500/20">
                    <div class="text-red-400 text-sm font-medium flex items-center gap-2"><i data-lucide="x-circle" class="w-4 h-4 inline-block align-middle"></i> Error</div>
                    <div class="text-red-300/80 text-xs mt-2 mono">{}</div>
                    <div class="text-zinc-600 text-xs mt-2 mono">Query: {}</div>
                </div>"##,
                html_escape(&e),
                html_escape(query),
            ))
        }
    }
}

/// Create web UI router
pub fn create_web_router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/", get(dashboard_page))
        .route("/memories", get(memories_page))
        .route("/visual", get(visual_page))
        .route("/mindmap", get(mindmap_page))
        .route("/mindmap/data", get(mindmap_data))
        .route("/timeline", get(timeline_page))
        .route("/timeline/data", get(timeline_data))
        .route("/coredb", get(coredb_page))
        .route("/coredb/query", axum::routing::post(coredb_query))
        .route("/search", get(search_page))
        .route("/search/results", axum::routing::post(search_results))
        .route("/store", get(store_page))
        .route("/store/submit", axum::routing::post(store_submit))
        .route("/api/visual/search", axum::routing::post(visual_search))
        .route("/api/visual/store", axum::routing::post(visual_store))
        .route("/api/visual/thumb", get(visual_thumb))
}

/// Escape HTML characters
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}
