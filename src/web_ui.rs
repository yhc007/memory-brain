//! Web UI for Memory Brain
//! 
//! Beautiful dashboard with HTMX for interactivity! 🌐

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
        r##"<h1 class="text-4xl font-bold mb-8">📊 Dashboard</h1>

<div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
    <div class="bg-gradient-to-br from-green-900/50 to-green-800/30 rounded-2xl p-6 border border-green-700/50 glow-green">
        <div class="text-green-400 text-sm">📥 Stores Today</div>
        <div class="text-4xl font-bold text-green-300 mt-2">{}</div>
    </div>
    
    <div class="bg-gradient-to-br from-blue-900/50 to-blue-800/30 rounded-2xl p-6 border border-blue-700/50">
        <div class="text-blue-400 text-sm">🔍 Recalls Today</div>
        <div class="text-4xl font-bold text-blue-300 mt-2">{}</div>
    </div>
    
    <div class="bg-gradient-to-br from-yellow-900/50 to-yellow-800/30 rounded-2xl p-6 border border-yellow-700/50">
        <div class="text-yellow-400 text-sm">🔎 Searches Today</div>
        <div class="text-4xl font-bold text-yellow-300 mt-2">{}</div>
    </div>
    
    <div class="bg-gradient-to-br from-cyan-900/50 to-cyan-800/30 rounded-2xl p-6 border border-cyan-700/50 glow">
        <div class="text-cyan-400 text-sm">🧠 Total Memories</div>
        <div class="text-4xl font-bold text-cyan-300 mt-2">{}</div>
    </div>
</div>

<div class="grid grid-cols-1 md:grid-cols-2 gap-6">
    <div class="bg-gray-800/50 rounded-2xl p-6 border border-gray-700">
        <h2 class="text-xl font-semibold mb-4">📈 Today's Activity</h2>
        <div class="space-y-3">
            <div class="flex items-center gap-3">
                <span class="text-green-400 w-24">Stores</span>
                <div class="flex-1 h-4 bg-gray-700 rounded-full overflow-hidden">
                    <div class="h-full bg-green-500 rounded-full" style="width: {}%"></div>
                </div>
                <span class="text-gray-400 w-12 text-right">{}</span>
            </div>
            <div class="flex items-center gap-3">
                <span class="text-blue-400 w-24">Recalls</span>
                <div class="flex-1 h-4 bg-gray-700 rounded-full overflow-hidden">
                    <div class="h-full bg-blue-500 rounded-full" style="width: {}%"></div>
                </div>
                <span class="text-gray-400 w-12 text-right">{}</span>
            </div>
            <div class="flex items-center gap-3">
                <span class="text-yellow-400 w-24">Searches</span>
                <div class="flex-1 h-4 bg-gray-700 rounded-full overflow-hidden">
                    <div class="h-full bg-yellow-500 rounded-full" style="width: {}%"></div>
                </div>
                <span class="text-gray-400 w-12 text-right">{}</span>
            </div>
        </div>
    </div>
    
    <div class="bg-gray-800/50 rounded-2xl p-6 border border-gray-700">
        <h2 class="text-xl font-semibold mb-4">⚡ Quick Actions</h2>
        <div class="flex flex-wrap gap-3">
            <a href="/store" class="bg-cyan-600 hover:bg-cyan-500 px-6 py-3 rounded-xl transition">
                ➕ Store Memory
            </a>
            <a href="/search" class="bg-purple-600 hover:bg-purple-500 px-6 py-3 rounded-xl transition">
                🔍 Search
            </a>
            <a href="/memories" class="bg-gray-600 hover:bg-gray-500 px-6 py-3 rounded-xl transition">
                📋 Browse All
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
            .map(|t| format!(r#"<span class="bg-cyan-900/50 text-cyan-400 px-2 py-1 rounded-lg text-sm">#{}</span>"#, t))
            .collect::<Vec<_>>()
            .join(" ");
        
        memory_cards.push_str(&format!(
            r##"<div class="bg-gray-800/50 rounded-2xl p-6 border border-gray-700 hover:border-cyan-700/50 transition">
                <p class="text-gray-200 mb-4">{}</p>
                <div class="flex justify-between items-center">
                    <div class="flex gap-2">{}</div>
                    <span class="text-gray-500 text-sm font-mono">{}</span>
                </div>
            </div>"##, 
            html_escape(&mem.content),
            tags_html,
            &mem.id.to_string()[..8]
        ));
    }
    
    let content = format!(
        r##"<div class="flex justify-between items-center mb-8">
            <h1 class="text-4xl font-bold">🧠 Memories</h1>
            <a href="/store" class="bg-cyan-600 hover:bg-cyan-500 px-6 py-3 rounded-xl transition">
                ➕ Store New
            </a>
        </div>
        <div class="space-y-4">{}</div>"##,
        if memory_cards.is_empty() { 
            r#"<div class="text-center text-gray-500 py-12">No memories yet. <a href="/store" class="text-cyan-400 hover:underline">Store your first memory!</a></div>"#.to_string()
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
pub struct SearchForm {
    query: String,
}

pub async fn search_results(
    State(state): State<Arc<AppState>>,
    Form(form): Form<SearchForm>,
) -> Html<String> {
    if form.query.is_empty() {
        return Html(r#"<div class="text-gray-500 text-center py-8">Enter a search query</div>"#.to_string());
    }
    
    let mut brain = state.brain.write().await;
    let memories = brain.recall(&form.query, 20);
    
    if memories.is_empty() {
        return Html(format!(
            r##"<div class="text-center py-12">
                <div class="text-4xl mb-4">🔍</div>
                <div class="text-gray-400">No memories found for "{}"</div>
            </div>"##, 
            html_escape(&form.query)
        ));
    }
    
    let mut html = String::new();
    for mem in memories {
        let tags_html: String = mem.tags.iter()
            .map(|t| format!(r#"<span class="bg-cyan-900/50 text-cyan-400 px-2 py-1 rounded-lg text-sm">#{}</span>"#, t))
            .collect::<Vec<_>>()
            .join(" ");
        
        html.push_str(&format!(
            r##"<div class="bg-gray-800/50 rounded-2xl p-6 border border-gray-700 hover:border-cyan-700/50 transition">
                <p class="text-gray-200 mb-4">{}</p>
                <div class="flex justify-between items-center">
                    <div class="flex gap-2">{}</div>
                    <span class="text-gray-500 text-sm font-mono">{}</span>
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
        return Html(r#"<div class="text-red-400 p-4 rounded-xl bg-red-900/30 border border-red-700">❌ Content is required</div>"#.to_string());
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
                r##"<div class="text-green-400 p-4 rounded-xl bg-green-900/30 border border-green-700">
                    ✅ Memory stored! ID: <span class="font-mono">{}</span>
                </div>
                <script>setTimeout(() => document.querySelector('form').reset(), 100);</script>"##,
                &item.id.to_string()[..8]
            ))
        }
        Err(e) => {
            Html(format!(
                r##"<div class="text-red-400 p-4 rounded-xl bg-red-900/30 border border-red-700">❌ Error: {}</div>"##,
                e
            ))
        }
    }
}

/// Create web UI router
pub fn create_web_router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/", get(dashboard_page))
        .route("/memories", get(memories_page))
        .route("/search", get(search_page))
        .route("/search/results", axum::routing::post(search_results))
        .route("/store", get(store_page))
        .route("/store/submit", axum::routing::post(store_submit))
}

/// Escape HTML characters
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}
