//! Constellation - Spatial Memory Visualization
//!
//! Projects memories into 2D space using PCA on embeddings,
//! creating a constellation-like view where similar memories cluster together.

use crate::{Brain, MemoryItem};
use std::collections::HashMap;

/// A point in 2D space representing a memory
#[derive(Debug, Clone)]
pub struct Star {
    pub id: String,
    pub x: f32,
    pub y: f32,
    pub label: String,
    pub content: String,
    pub tags: Vec<String>,
    pub strength: f32,
    pub cluster: usize,
}

/// Constellation visualization data
#[derive(Debug)]
pub struct Constellation {
    pub stars: Vec<Star>,
    pub clusters: Vec<String>, // cluster names (primary tags)
}

impl Constellation {
    /// Build constellation from brain memories using PCA projection
    pub fn from_brain(brain: &Brain, limit: usize) -> Self {
        let mut memories: Vec<MemoryItem> = Vec::new();
        
        // Gather memories with embeddings
        if let Ok(items) = brain.semantic.search("", limit) {
            for item in items {
                if item.embedding.is_some() {
                    memories.push(item);
                }
            }
        }
        
        if memories.is_empty() {
            return Self { stars: vec![], clusters: vec![] };
        }
        
        // Extract embeddings matrix
        let embeddings: Vec<&Vec<f32>> = memories.iter()
            .filter_map(|m| m.embedding.as_ref())
            .collect();
        
        // Apply PCA to reduce to 2D
        let coords = pca_2d(&embeddings);
        
        // Assign clusters based on primary tag
        let mut tag_to_cluster: HashMap<String, usize> = HashMap::new();
        let mut cluster_names: Vec<String> = Vec::new();
        
        let mut stars = Vec::new();
        
        for (i, memory) in memories.iter().enumerate() {
            let primary_tag = memory.tags.first()
                .cloned()
                .unwrap_or_else(|| "general".to_string());
            
            let cluster = *tag_to_cluster.entry(primary_tag.clone()).or_insert_with(|| {
                let idx = cluster_names.len();
                cluster_names.push(primary_tag.clone());
                idx
            });
            
            let (x, y) = if i < coords.len() {
                coords[i]
            } else {
                (0.0, 0.0)
            };
            
            stars.push(Star {
                id: memory.id.to_string(),
                label: truncate(&memory.content, 25),
                content: memory.content.clone(),
                x,
                y,
                tags: memory.tags.clone(),
                strength: memory.strength,
                cluster,
            });
        }
        
        Self { stars, clusters: cluster_names }
    }
    
    /// Generate interactive HTML visualization
    pub fn to_html(&self) -> String {
        let stars_json = self.stars_to_json();
        let clusters_json = self.clusters_to_json();
        
        format!(r##"<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>🌌 Memory Constellation</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: radial-gradient(ellipse at center, #1a1a2e 0%, #0d0d1a 100%);
            overflow: hidden;
            color: #fff;
        }}
        #container {{
            width: 100vw;
            height: 100vh;
        }}
        .star {{
            cursor: pointer;
            transition: all 0.3s;
        }}
        .star:hover {{
            filter: brightness(1.5);
        }}
        .star.selected {{
            filter: brightness(2) drop-shadow(0 0 20px currentColor);
        }}
        .star-label {{
            font-size: 10px;
            fill: rgba(255,255,255,0.9);
            pointer-events: none;
            text-shadow: 0 0 5px rgba(0,0,0,0.8);
            opacity: 0;
            transition: opacity 0.2s;
        }}
        .star:hover .star-label {{
            opacity: 1;
        }}
        .star.selected .star-label {{
            opacity: 1;
        }}
        .show-labels .star-label {{
            opacity: 0.7;
        }}
        #title {{
            position: absolute;
            top: 20px;
            left: 20px;
            font-size: 28px;
            font-weight: bold;
            text-shadow: 0 0 20px rgba(255,255,255,0.3);
        }}
        #search-box {{
            position: absolute;
            top: 70px;
            left: 20px;
            z-index: 100;
        }}
        #search-input {{
            width: 280px;
            padding: 12px 18px;
            font-size: 14px;
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 30px;
            background: rgba(0,0,0,0.4);
            color: #fff;
            outline: none;
            backdrop-filter: blur(10px);
        }}
        #search-input::placeholder {{
            color: rgba(255,255,255,0.4);
        }}
        #search-input:focus {{
            border-color: rgba(255,255,255,0.5);
            box-shadow: 0 0 30px rgba(255,255,255,0.1);
        }}
        #search-results {{
            margin-top: 10px;
            font-size: 12px;
            color: rgba(255,255,255,0.5);
        }}
        #tooltip {{
            position: absolute;
            background: rgba(0,0,0,0.9);
            border: 1px solid rgba(255,255,255,0.2);
            padding: 15px;
            border-radius: 12px;
            font-size: 13px;
            max-width: 350px;
            pointer-events: none;
            display: none;
            backdrop-filter: blur(10px);
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        }}
        #tooltip .title {{
            font-weight: bold;
            margin-bottom: 8px;
            color: #fff;
        }}
        #tooltip .content {{
            color: rgba(255,255,255,0.8);
            line-height: 1.5;
        }}
        #tooltip .tags {{
            margin-top: 10px;
        }}
        #tooltip .tag {{
            display: inline-block;
            background: rgba(255,255,255,0.15);
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 11px;
            margin-right: 5px;
            margin-bottom: 5px;
        }}
        #legend {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(0,0,0,0.5);
            padding: 15px 20px;
            border-radius: 12px;
            backdrop-filter: blur(10px);
        }}
        #legend h3 {{
            font-size: 12px;
            margin-bottom: 10px;
            color: rgba(255,255,255,0.6);
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 6px 0;
            font-size: 12px;
            cursor: pointer;
            opacity: 0.8;
            transition: opacity 0.2s;
        }}
        .legend-item:hover {{
            opacity: 1;
        }}
        .legend-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 10px;
        }}
        #stats {{
            position: absolute;
            bottom: 20px;
            right: 20px;
            font-size: 12px;
            color: rgba(255,255,255,0.4);
        }}
        #instructions {{
            position: absolute;
            top: 20px;
            right: 20px;
            font-size: 11px;
            color: rgba(255,255,255,0.4);
            text-align: right;
            line-height: 1.8;
        }}
        .dimmed {{
            opacity: 0.1 !important;
        }}
    </style>
</head>
<body>
    <div id="container"></div>
    <div id="title">🌌 Memory Constellation</div>
    <div id="search-box">
        <input type="text" id="search-input" placeholder="🔍 기억 검색..." />
        <div id="search-results"></div>
    </div>
    <div id="tooltip">
        <div class="title"></div>
        <div class="content"></div>
        <div class="tags"></div>
    </div>
    <div id="legend">
        <h3>CLUSTERS</h3>
    </div>
    <div id="stats">{} memories</div>
    <div id="instructions">
        🖱️ 드래그: 이동<br>
        🔍 스크롤: 줌<br>
        👆 클릭: 선택<br>
        ⎋ ESC: 초기화<br>
        <button id="toggle-labels" style="margin-top:10px;padding:5px 10px;border-radius:15px;border:1px solid rgba(255,255,255,0.3);background:rgba(0,0,0,0.3);color:#fff;cursor:pointer;font-size:11px;">📝 라벨 표시</button>
    </div>
    
    <script>
        const stars = {stars_json};
        const clusterNames = {clusters_json};
        const colors = d3.schemeSet2;
        
        const width = window.innerWidth;
        const height = window.innerHeight;
        
        // Scale coordinates to fit screen
        const xExtent = d3.extent(stars, d => d.x);
        const yExtent = d3.extent(stars, d => d.y);
        
        const xScale = d3.scaleLinear()
            .domain(xExtent)
            .range([100, width - 100]);
        
        const yScale = d3.scaleLinear()
            .domain(yExtent)
            .range([100, height - 100]);
        
        const svg = d3.select("#container")
            .append("svg")
            .attr("width", width)
            .attr("height", height);
        
        // Add stars background effect
        const defs = svg.append("defs");
        
        const glow = defs.append("filter")
            .attr("id", "glow");
        glow.append("feGaussianBlur")
            .attr("stdDeviation", "2")
            .attr("result", "coloredBlur");
        const feMerge = glow.append("feMerge");
        feMerge.append("feMergeNode").attr("in", "coloredBlur");
        feMerge.append("feMergeNode").attr("in", "SourceGraphic");
        
        // Background stars
        const bgStars = svg.append("g").attr("class", "bg-stars");
        for (let i = 0; i < 200; i++) {{
            bgStars.append("circle")
                .attr("cx", Math.random() * width)
                .attr("cy", Math.random() * height)
                .attr("r", Math.random() * 1.5)
                .attr("fill", `rgba(255,255,255,${{Math.random() * 0.5 + 0.1}})`);
        }}
        
        const g = svg.append("g");
        
        // Zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.3, 5])
            .on("zoom", (event) => {{
                g.attr("transform", event.transform);
            }});
        
        svg.call(zoom);
        
        // Draw memory stars
        const star = g.selectAll(".star")
            .data(stars)
            .join("g")
            .attr("class", "star")
            .attr("transform", d => `translate(${{xScale(d.x)}},${{yScale(d.y)}})`);
        
        star.append("circle")
            .attr("r", d => Math.max(4, d.strength * 8 + 3))
            .attr("fill", d => colors[d.cluster % colors.length])
            .attr("filter", "url(#glow)");
        
        star.append("text")
            .attr("class", "star-label")
            .attr("dy", d => -Math.max(4, d.strength * 8 + 3) - 5)
            .attr("text-anchor", "middle")
            .text(d => d.label);
        
        // Tooltip
        const tooltip = d3.select("#tooltip");
        
        star.on("mouseover", (event, d) => {{
            tooltip.style("display", "block")
                .style("left", (event.pageX + 20) + "px")
                .style("top", (event.pageY - 10) + "px");
            tooltip.select(".title").text(d.label);
            tooltip.select(".content").text(d.content);
            tooltip.select(".tags").html(d.tags.map(t => `<span class="tag">#${{t}}</span>`).join(""));
        }})
        .on("mousemove", (event) => {{
            tooltip.style("left", (event.pageX + 20) + "px")
                .style("top", (event.pageY - 10) + "px");
        }})
        .on("mouseout", () => {{
            tooltip.style("display", "none");
        }});
        
        // Selection
        let selectedStar = null;
        
        star.on("click", (event, d) => {{
            event.stopPropagation();
            
            if (selectedStar === d) {{
                // Deselect
                selectedStar = null;
                star.classed("selected", false).classed("dimmed", false);
                searchResults.textContent = "";
            }} else {{
                selectedStar = d;
                star.classed("selected", s => s.id === d.id)
                    .classed("dimmed", s => s.id !== d.id && s.cluster !== d.cluster);
                searchResults.textContent = `"${{d.label}}" - ${{clusterNames[d.cluster] || 'cluster ' + d.cluster}}`;
                
                // Zoom to selection
                const x = xScale(d.x);
                const y = yScale(d.y);
                svg.transition().duration(750).call(
                    zoom.transform,
                    d3.zoomIdentity.translate(width/2 - x, height/2 - y).scale(1.5)
                );
            }}
        }});
        
        svg.on("click", () => {{
            selectedStar = null;
            star.classed("selected", false).classed("dimmed", false);
            searchResults.textContent = "";
            svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity);
        }});
        
        // Search
        const searchInput = document.getElementById("search-input");
        const searchResults = document.getElementById("search-results");
        
        let searchTimeout;
        searchInput.addEventListener("input", (e) => {{
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {{
                const query = e.target.value.toLowerCase().trim();
                
                if (!query) {{
                    star.classed("dimmed", false);
                    searchResults.textContent = "";
                    return;
                }}
                
                let matchCount = 0;
                star.classed("dimmed", d => {{
                    const match = (d.content + " " + d.tags.join(" ")).toLowerCase().includes(query);
                    if (match) matchCount++;
                    return !match;
                }});
                
                searchResults.textContent = `${{matchCount}}개 발견`;
            }}, 150);
        }});
        
        // Keyboard
        document.addEventListener("keydown", (e) => {{
            if (e.key === "Escape") {{
                searchInput.value = "";
                selectedStar = null;
                star.classed("selected", false).classed("dimmed", false);
                searchResults.textContent = "";
                svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
            }}
            if ((e.ctrlKey || e.metaKey) && e.key === "f") {{
                e.preventDefault();
                searchInput.focus();
            }}
        }});
        
        // Legend
        const legend = d3.select("#legend");
        clusterNames.slice(0, 10).forEach((name, i) => {{
            const item = legend.append("div")
                .attr("class", "legend-item")
                .on("click", () => {{
                    star.classed("dimmed", d => d.cluster !== i);
                    searchResults.textContent = `Cluster: ${{name}}`;
                }});
            
            item.append("div")
                .attr("class", "legend-dot")
                .style("background", colors[i % colors.length]);
            
            item.append("span").text(name);
        }});
        
        // Toggle labels button
        const toggleBtn = document.getElementById("toggle-labels");
        let labelsVisible = false;
        
        toggleBtn.addEventListener("click", () => {{
            labelsVisible = !labelsVisible;
            g.classed("show-labels", labelsVisible);
            toggleBtn.textContent = labelsVisible ? "📝 라벨 숨기기" : "📝 라벨 표시";
        }});
        
        // Show labels when zoomed in
        let currentZoom = 1;
        svg.call(zoom.on("zoom", (event) => {{
            g.attr("transform", event.transform);
            currentZoom = event.transform.k;
            
            // Auto-show labels when zoomed in past 1.5x
            if (currentZoom > 1.5 && !labelsVisible) {{
                g.classed("show-labels", true);
            }} else if (currentZoom <= 1.5 && !labelsVisible) {{
                g.classed("show-labels", false);
            }}
        }}));
    </script>
</body>
</html>"##, self.stars.len(), stars_json = stars_json, clusters_json = clusters_json)
    }
    
    fn stars_to_json(&self) -> String {
        let items: Vec<String> = self.stars.iter().map(|s| {
            format!(
                r#"{{"id":"{}","x":{},"y":{},"label":"{}","content":"{}","tags":[{}],"strength":{},"cluster":{}}}"#,
                s.id,
                s.x,
                s.y,
                escape_json(&s.label),
                escape_json(&s.content),
                s.tags.iter().map(|t| format!("\"{}\"", escape_json(t))).collect::<Vec<_>>().join(","),
                s.strength,
                s.cluster
            )
        }).collect();
        format!("[{}]", items.join(","))
    }
    
    fn clusters_to_json(&self) -> String {
        let items: Vec<String> = self.clusters.iter()
            .map(|c| format!("\"{}\"", escape_json(c)))
            .collect();
        format!("[{}]", items.join(","))
    }
}

/// Simple PCA for 2D projection
fn pca_2d(embeddings: &[&Vec<f32>]) -> Vec<(f32, f32)> {
    if embeddings.is_empty() {
        return vec![];
    }
    
    let n = embeddings.len();
    let dim = embeddings[0].len();
    
    if dim == 0 {
        return vec![(0.0, 0.0); n];
    }
    
    // Filter to only embeddings with matching dimensions
    let valid_embeddings: Vec<&Vec<f32>> = embeddings.iter()
        .filter(|e| e.len() == dim)
        .copied()
        .collect();
    
    if valid_embeddings.is_empty() {
        return vec![(0.0, 0.0); n];
    }
    
    // Calculate mean
    let mut mean = vec![0.0f32; dim];
    for emb in &valid_embeddings {
        for (i, &v) in emb.iter().enumerate() {
            if i < dim {
                mean[i] += v;
            }
        }
    }
    let count = valid_embeddings.len() as f32;
    for m in &mut mean {
        *m /= count;
    }
    
    // Center the data
    let centered: Vec<Vec<f32>> = valid_embeddings.iter()
        .map(|emb| {
            emb.iter()
                .take(dim)
                .zip(&mean)
                .map(|(&v, &m)| v - m)
                .collect()
        })
        .collect();
    
    // Power iteration for first two principal components
    let pc1 = power_iteration(&centered, None, 30);
    let pc2 = power_iteration(&centered, Some(&pc1), 30);
    
    // Project all original embeddings onto principal components
    // Use zero for embeddings with mismatched dimensions
    embeddings.iter().map(|emb| {
        if emb.len() != dim {
            return (0.0, 0.0);
        }
        // Center this embedding
        let centered_emb: Vec<f32> = emb.iter()
            .take(dim)
            .zip(&mean)
            .map(|(&v, &m)| v - m)
            .collect();
        
        let x: f32 = centered_emb.iter().zip(&pc1).map(|(&a, &b)| a * b).sum();
        let y: f32 = centered_emb.iter().zip(&pc2).map(|(&a, &b)| a * b).sum();
        (x, y)
    }).collect()
}

/// Power iteration to find principal component
fn power_iteration(data: &[Vec<f32>], deflate: Option<&Vec<f32>>, iterations: usize) -> Vec<f32> {
    if data.is_empty() {
        return vec![];
    }
    
    let dim = data[0].len();
    if dim == 0 {
        return vec![];
    }
    
    // Initialize with pseudo-random vector
    let mut v: Vec<f32> = (0..dim).map(|i| {
        let x = ((i * 7 + 3) % 100) as f32 / 100.0;
        if x == 0.0 { 0.01 } else { x }
    }).collect();
    
    // Normalize
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in &mut v {
            *x /= norm;
        }
    }
    
    for _ in 0..iterations {
        // Compute A^T * A * v (covariance matrix times v)
        let mut new_v = vec![0.0f32; dim];
        
        for row in data {
            if row.len() != dim {
                continue;
            }
            let dot: f32 = row.iter().zip(&v).map(|(&a, &b)| a * b).sum();
            for (i, &r) in row.iter().enumerate() {
                if i < dim {
                    new_v[i] += dot * r;
                }
            }
        }
        
        // Deflate if finding second component
        if let Some(pc1) = deflate {
            if pc1.len() == dim {
                let dot: f32 = new_v.iter().zip(pc1).map(|(&a, &b)| a * b).sum();
                for (i, &p) in pc1.iter().enumerate() {
                    if i < dim {
                        new_v[i] -= dot * p;
                    }
                }
            }
        }
        
        // Normalize
        let norm: f32 = new_v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in &mut new_v {
                *x /= norm;
            }
        }
        
        v = new_v;
    }
    
    v
}

fn truncate(s: &str, max: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max {
        s.to_string()
    } else {
        chars[..max].iter().collect::<String>() + "..."
    }
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pca_basic() {
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let refs: Vec<&Vec<f32>> = embeddings.iter().collect();
        let coords = pca_2d(&refs);
        assert_eq!(coords.len(), 3);
    }
}
