//! Mind Map - Interactive Memory Visualization
//!
//! Generates an interactive HTML visualization of memory connections
//! using D3.js force-directed graph.

use crate::{Brain, MemoryItem, cosine_similarity};
use std::collections::{HashMap, HashSet};

/// Node in the mind map
#[derive(Debug, Clone)]
pub struct MapNode {
    pub id: String,
    pub label: String,
    pub content: String,
    pub group: usize,
    pub size: f32,
    pub tags: Vec<String>,
}

/// Edge connecting two nodes
#[derive(Debug, Clone)]
pub struct MapEdge {
    pub source: String,
    pub target: String,
    pub weight: f32,
}

/// Mind map data structure
#[derive(Debug)]
pub struct MindMap {
    pub nodes: Vec<MapNode>,
    pub edges: Vec<MapEdge>,
}

impl MindMap {
    /// Build mind map from brain memories
    pub fn from_brain(brain: &Brain, limit: usize, threshold: f32) -> Self {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut memories: Vec<MemoryItem> = Vec::new();

        // Gather memories
        if let Ok(items) = brain.semantic.search("", limit) {
            memories.extend(items);
        }

        if memories.is_empty() {
            return Self { nodes, edges };
        }

        // Assign groups based on tags
        let mut tag_groups: HashMap<String, usize> = HashMap::new();
        let mut next_group = 0;

        // Create nodes
        for memory in &memories {
            let primary_tag = memory.tags.first().cloned().unwrap_or_else(|| "general".to_string());
            
            let group = *tag_groups.entry(primary_tag.clone()).or_insert_with(|| {
                let g = next_group;
                next_group += 1;
                g
            });

            let label = truncate(&memory.content, 20);
            
            nodes.push(MapNode {
                id: memory.id.to_string(),
                label,
                content: memory.content.clone(),
                group,
                size: (memory.strength * 10.0 + 5.0).min(20.0),
                tags: memory.tags.clone(),
            });
        }

        // Create edges based on similarity
        let mut seen_pairs: HashSet<(String, String)> = HashSet::new();
        
        for i in 0..memories.len() {
            for j in (i + 1)..memories.len() {
                if let (Some(emb_a), Some(emb_b)) = (&memories[i].embedding, &memories[j].embedding) {
                    let sim = cosine_similarity(emb_a, emb_b);
                    
                    if sim > threshold {
                        let id_a = memories[i].id.to_string();
                        let id_b = memories[j].id.to_string();
                        
                        let pair = if id_a < id_b {
                            (id_a.clone(), id_b.clone())
                        } else {
                            (id_b.clone(), id_a.clone())
                        };
                        
                        if !seen_pairs.contains(&pair) {
                            seen_pairs.insert(pair);
                            edges.push(MapEdge {
                                source: id_a,
                                target: id_b,
                                weight: sim,
                            });
                        }
                    }
                }
            }
        }

        Self { nodes, edges }
    }

    /// Generate interactive HTML
    pub fn to_html(&self) -> String {
        let nodes_json = self.nodes_to_json();
        let edges_json = self.edges_to_json();
        
        format!(r##"<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>🧠 Memory Mind Map</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            overflow: hidden;
        }}
        #container {{
            width: 100vw;
            height: 100vh;
        }}
        .node {{
            cursor: pointer;
        }}
        .node circle {{
            stroke: #fff;
            stroke-width: 2px;
            transition: all 0.3s;
        }}
        .node:hover circle {{
            stroke-width: 4px;
            filter: brightness(1.3);
        }}
        .node text {{
            fill: #fff;
            font-size: 10px;
            pointer-events: none;
            text-shadow: 0 0 3px rgba(0,0,0,0.8);
        }}
        .link {{
            stroke: rgba(255,255,255,0.2);
            stroke-width: 1px;
        }}
        .link:hover {{
            stroke: rgba(255,255,255,0.6);
        }}
        #tooltip {{
            position: absolute;
            background: rgba(0,0,0,0.9);
            color: #fff;
            padding: 12px;
            border-radius: 8px;
            font-size: 13px;
            max-width: 300px;
            pointer-events: none;
            display: none;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            border: 1px solid rgba(255,255,255,0.1);
        }}
        #tooltip .tags {{
            margin-top: 8px;
            font-size: 11px;
            color: #888;
        }}
        #tooltip .tags span {{
            background: #333;
            padding: 2px 6px;
            border-radius: 4px;
            margin-right: 4px;
        }}
        #title {{
            position: absolute;
            top: 20px;
            left: 20px;
            color: #fff;
            font-size: 24px;
            font-weight: bold;
            text-shadow: 0 2px 10px rgba(0,0,0,0.5);
        }}
        #stats {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            color: rgba(255,255,255,0.6);
            font-size: 12px;
        }}
        #legend {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(0,0,0,0.5);
            padding: 15px;
            border-radius: 8px;
            color: #fff;
            font-size: 12px;
        }}
        #legend .item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
        }}
        #legend .dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
    </style>
</head>
<body>
    <div id="container"></div>
    <div id="title">🧠 Memory Mind Map</div>
    <div id="stats">{} nodes, {} connections</div>
    <div id="tooltip"></div>
    <div id="legend"></div>
    
    <script>
        const nodes = {nodes_json};
        const links = {edges_json};
        
        const colors = d3.schemeCategory10;
        
        const width = window.innerWidth;
        const height = window.innerHeight;
        
        const svg = d3.select("#container")
            .append("svg")
            .attr("width", width)
            .attr("height", height);
        
        // Add zoom behavior
        const g = svg.append("g");
        
        svg.call(d3.zoom()
            .scaleExtent([0.1, 4])
            .on("zoom", (event) => {{
                g.attr("transform", event.transform);
            }}));
        
        // Create simulation
        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-200))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(d => d.size + 10));
        
        // Draw links
        const link = g.append("g")
            .selectAll("line")
            .data(links)
            .join("line")
            .attr("class", "link")
            .attr("stroke-width", d => d.weight * 3);
        
        // Draw nodes
        const node = g.append("g")
            .selectAll("g")
            .data(nodes)
            .join("g")
            .attr("class", "node")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));
        
        node.append("circle")
            .attr("r", d => d.size)
            .attr("fill", d => colors[d.group % 10]);
        
        node.append("text")
            .attr("dx", d => d.size + 5)
            .attr("dy", 4)
            .text(d => d.label);
        
        // Tooltip
        const tooltip = d3.select("#tooltip");
        
        node.on("mouseover", (event, d) => {{
            tooltip.style("display", "block")
                .html(`<strong>${{d.label}}</strong><br>${{d.content}}<div class="tags">${{d.tags.map(t => `<span>#${{t}}</span>`).join('')}}</div>`)
                .style("left", (event.pageX + 15) + "px")
                .style("top", (event.pageY - 10) + "px");
        }})
        .on("mouseout", () => {{
            tooltip.style("display", "none");
        }});
        
        // Update positions
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            
            node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
        }});
        
        // Drag functions
        function dragstarted(event) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }}
        
        function dragged(event) {{
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }}
        
        function dragended(event) {{
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }}
        
        // Build legend
        const groups = [...new Set(nodes.map(n => n.group))];
        const tagNames = {{}};
        nodes.forEach(n => {{
            if (n.tags.length > 0 && !tagNames[n.group]) {{
                tagNames[n.group] = n.tags[0];
            }}
        }});
        
        const legend = d3.select("#legend");
        groups.slice(0, 8).forEach(g => {{
            legend.append("div")
                .attr("class", "item")
                .html(`<div class="dot" style="background:${{colors[g % 10]}}"></div>${{tagNames[g] || 'Group ' + g}}`);
        }});
    </script>
</body>
</html>"##, self.nodes.len(), self.edges.len(), nodes_json = nodes_json, edges_json = edges_json)
    }

    fn nodes_to_json(&self) -> String {
        let items: Vec<String> = self.nodes.iter().map(|n| {
            format!(
                r#"{{"id":"{}","label":"{}","content":"{}","group":{},"size":{},"tags":[{}]}}"#,
                n.id,
                escape_json(&n.label),
                escape_json(&n.content),
                n.group,
                n.size,
                n.tags.iter().map(|t| format!("\"{}\"", escape_json(t))).collect::<Vec<_>>().join(",")
            )
        }).collect();
        format!("[{}]", items.join(","))
    }

    fn edges_to_json(&self) -> String {
        let items: Vec<String> = self.edges.iter().map(|e| {
            format!(
                r#"{{"source":"{}","target":"{}","weight":{:.3}}}"#,
                e.source, e.target, e.weight
            )
        }).collect();
        format!("[{}]", items.join(","))
    }

    /// Generate DOT format for Graphviz
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph MindMap {\n");
        dot.push_str("  rankdir=LR;\n");
        dot.push_str("  node [shape=box, style=rounded];\n\n");
        
        for node in &self.nodes {
            dot.push_str(&format!(
                "  \"{}\" [label=\"{}\"];\n",
                node.id, escape_dot(&node.label)
            ));
        }
        
        dot.push_str("\n");
        
        for edge in &self.edges {
            dot.push_str(&format!(
                "  \"{}\" -> \"{}\" [weight={:.2}];\n",
                edge.source, edge.target, edge.weight
            ));
        }
        
        dot.push_str("}\n");
        dot
    }

    /// Generate Mermaid format
    pub fn to_mermaid(&self) -> String {
        let mut mermaid = String::from("graph LR\n");
        
        for (i, node) in self.nodes.iter().enumerate() {
            mermaid.push_str(&format!(
                "  {}[\"{}\"]\n",
                i, escape_mermaid(&node.label)
            ));
        }
        
        // Create id to index map
        let id_map: HashMap<&str, usize> = self.nodes.iter()
            .enumerate()
            .map(|(i, n)| (n.id.as_str(), i))
            .collect();
        
        for edge in &self.edges {
            if let (Some(&src), Some(&tgt)) = (id_map.get(edge.source.as_str()), id_map.get(edge.target.as_str())) {
                mermaid.push_str(&format!("  {} --> {}\n", src, tgt));
            }
        }
        
        mermaid
    }
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

fn escape_dot(s: &str) -> String {
    s.replace('"', "\\\"").replace('\n', " ")
}

fn escape_mermaid(s: &str) -> String {
    s.replace('"', "'").replace('\n', " ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_mindmap_creation() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("map_test.db");
        let mut brain = Brain::new(db_path.to_str().unwrap()).unwrap();
        
        // Add memories (use "is/are" to classify as Semantic)
        brain.process("Rust is a systems programming language", None).unwrap();
        brain.process("Ownership is a core concept in Rust", None).unwrap();
        brain.process("Python is a scripting language", None).unwrap();
        
        let map = MindMap::from_brain(&brain, 100, 0.3);
        
        // Should have at least the 3 memories we added
        assert!(map.nodes.len() >= 3, "Expected >= 3 nodes, got {}", map.nodes.len());
    }

    #[test]
    fn test_html_generation() {
        let map = MindMap {
            nodes: vec![
                MapNode {
                    id: "1".to_string(),
                    label: "Test".to_string(),
                    content: "Test content".to_string(),
                    group: 0,
                    size: 10.0,
                    tags: vec!["test".to_string()],
                },
            ],
            edges: vec![],
        };
        
        let html = map.to_html();
        assert!(html.contains("Memory Mind Map"));
        assert!(html.contains("d3.") && html.contains(".js"));
    }
}
