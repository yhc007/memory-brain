//! Storage - CoreDB backend for persistent memory (sync wrapper)

use crate::types::{MemoryItem, MemoryType, Emotion};
use chrono::{DateTime, Utc};
use coredb::{CoreDB, DatabaseConfig};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::runtime::{Runtime, Handle};
use tokio::sync::RwLock;
use uuid::Uuid;

pub struct Storage {
    db: Arc<RwLock<CoreDB>>,
    /// Some if we own the runtime, None if reusing existing
    owned_runtime: Option<Runtime>,
    keyspace: String,
    table: String,
}

impl Storage {
    /// Run async code, reusing existing runtime if available
    fn block_on<F: std::future::Future>(&self, f: F) -> F::Output {
        if let Some(ref rt) = self.owned_runtime {
            rt.block_on(f)
        } else {
            // We're inside an existing runtime, use block_in_place
            tokio::task::block_in_place(|| {
                Handle::current().block_on(f)
            })
        }
    }

    pub fn new(db_path: &str, table_name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // Check if we're already in a tokio runtime
        let in_runtime = Handle::try_current().is_ok();
        let owned_runtime = if in_runtime { None } else { Some(Runtime::new()?) };
        
        let config = DatabaseConfig {
            data_directory: PathBuf::from(db_path).join("data"),
            commitlog_directory: PathBuf::from(db_path).join("commitlog"),
            memtable_flush_threshold_mb: 16,
            compaction_throughput_mb_per_sec: 16,
            concurrent_reads: 32,
            concurrent_writes: 32,
        };

        let db = if in_runtime {
            // Already in async context - use block_in_place
            tokio::task::block_in_place(|| {
                Handle::current().block_on(CoreDB::new(config))
            })?
        } else {
            // Not in async context - use our runtime
            owned_runtime.as_ref().unwrap().block_on(CoreDB::new(config))?
        };
        let keyspace = "memory_brain".to_string();
        
        let storage = Self {
            db: Arc::new(RwLock::new(db)),
            owned_runtime,
            keyspace: keyspace.clone(),
            table: table_name.to_string(),
        };

        storage.init_tables()?;
        Ok(storage)
    }

    fn init_tables(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.block_on(async {
            let db = self.db.read().await;

            // Create keyspace
            let ks_query = format!(
                "CREATE KEYSPACE {} WITH REPLICATION = {{'class': 'SimpleStrategy', 'replication_factor': 1}}",
                self.keyspace
            );
            let _ = db.execute_cql(&ks_query).await; // Ignore if exists

            // Create memories table
            let table_query = format!(
                "CREATE TABLE {}.{} (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    context TEXT,
                    memory_type TEXT,
                    emotion TEXT,
                    created_at BIGINT,
                    last_accessed BIGINT,
                    access_count INT,
                    strength TEXT,
                    embedding TEXT,
                    tags TEXT
                )",
                self.keyspace, self.table
            );
            let _ = db.execute_cql(&table_query).await; // Ignore if exists

            Ok(())
        })
    }

    /// Save a memory item
    pub fn save(&self, item: &MemoryItem) -> Result<(), Box<dyn std::error::Error>> {
        self.block_on(async {
            let db = self.db.read().await;

            let embedding_json = item.embedding.as_ref()
                .map(|e| serde_json::to_string(e).unwrap_or_default())
                .unwrap_or_default();
            
            let tags_json = serde_json::to_string(&item.tags)?;
            let context = item.context.clone().unwrap_or_default();

            // Escape single quotes for CQL
            let content = item.content.replace('\'', "''");
            let context = context.replace('\'', "''");
            let embedding_json = embedding_json.replace('\'', "''");
            let tags_json = tags_json.replace('\'', "''");

            let query = format!(
                "INSERT INTO {}.{} (id, content, context, memory_type, emotion, created_at, last_accessed, access_count, strength, embedding, tags) \
                 VALUES ('{}', '{}', '{}', '{}', '{}', {}, {}, {}, '{}', '{}', '{}')",
                self.keyspace, self.table,
                item.id,
                content,
                context,
                format!("{:?}", item.memory_type),
                format!("{:?}", item.emotion),
                item.created_at.timestamp_millis(),
                item.last_accessed.timestamp_millis(),
                item.access_count,
                item.strength,
                embedding_json,
                tags_json
            );

            db.execute_cql(&query).await?;
            
            // Flush immediately to persist data
            db.flush_all().await?;
            
            Ok(())
        })
    }

    /// Update a memory item
    pub fn update(&self, item: &MemoryItem) -> Result<(), Box<dyn std::error::Error>> {
        self.save(item)
    }

    /// Delete a memory item
    pub fn delete(&self, id: &Uuid) -> Result<(), Box<dyn std::error::Error>> {
        self.block_on(async {
            let db = self.db.read().await;
            let query = format!(
                "DELETE FROM {}.{} WHERE id = '{}'",
                self.keyspace, self.table, id
            );
            db.execute_cql(&query).await?;
            Ok(())
        })
    }

    /// Search memories by content
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryItem>, Box<dyn std::error::Error>> {
        self.block_on(async {
            let db = self.db.read().await;

            // CoreDB: fetch all and filter (TODO: add LIKE support to CoreDB)
            let cql = format!("SELECT * FROM {}.{}", self.keyspace, self.table);
            let result = db.execute_cql(&cql).await?;
            
            let mut items = self.parse_query_result(result)?;

            // Filter by query (simple substring match)
            if !query.is_empty() {
                let query_lower = query.to_lowercase();
                items.retain(|item| {
                    item.content.to_lowercase().contains(&query_lower) ||
                    item.context.as_ref().map(|c| c.to_lowercase().contains(&query_lower)).unwrap_or(false)
                });
            }

            // Sort by strength and recency
            items.sort_by(|a, b| {
                b.strength.partial_cmp(&a.strength).unwrap_or(std::cmp::Ordering::Equal)
            });

            items.truncate(limit);
            Ok(items)
        })
    }

    /// Get all memories
    pub fn get_all(&self) -> Result<Vec<MemoryItem>, Box<dyn std::error::Error>> {
        self.block_on(async {
            let db = self.db.read().await;
            let cql = format!("SELECT * FROM {}.{}", self.keyspace, self.table);
            let result = db.execute_cql(&cql).await?;
            self.parse_query_result(result)
        })
    }

    /// Get recent memories
    pub fn get_recent(&self, limit: usize) -> Result<Vec<MemoryItem>, Box<dyn std::error::Error>> {
        self.block_on(async {
            let db = self.db.read().await;
            let cql = format!("SELECT * FROM {}.{}", self.keyspace, self.table);
            let result = db.execute_cql(&cql).await?;
            let mut items = self.parse_query_result(result)?;
            
            // Sort by created_at DESC
            items.sort_by(|a, b| b.created_at.cmp(&a.created_at));
            items.truncate(limit);
            Ok(items)
        })
    }

    /// Get memories by time range
    pub fn get_by_time_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<MemoryItem>, Box<dyn std::error::Error>> {
        self.block_on(async {
            let db = self.db.read().await;
            let cql = format!("SELECT * FROM {}.{}", self.keyspace, self.table);
            let result = db.execute_cql(&cql).await?;
            let mut items = self.parse_query_result(result)?;
            
            // Filter by time range
            items.retain(|item| item.created_at >= start && item.created_at <= end);
            items.sort_by(|a, b| b.created_at.cmp(&a.created_at));
            Ok(items)
        })
    }

    /// Get memories by tag
    pub fn get_by_tag(&self, tag: &str) -> Result<Vec<MemoryItem>, Box<dyn std::error::Error>> {
        self.block_on(async {
            let db = self.db.read().await;
            let cql = format!("SELECT * FROM {}.{}", self.keyspace, self.table);
            let result = db.execute_cql(&cql).await?;
            let mut items = self.parse_query_result(result)?;
            
            // Filter by tag
            let tag_lower = tag.to_lowercase();
            items.retain(|item| {
                item.tags.iter().any(|t| t.to_lowercase().contains(&tag_lower))
            });
            items.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap_or(std::cmp::Ordering::Equal));
            Ok(items)
        })
    }

    /// Add association between memories
    /// Execute arbitrary CQL query and return HTML-formatted results
    pub fn execute_cql_html(&self, query: &str) -> Result<String, String> {
        use tokio::runtime::Handle;
        
        let db = self.db.clone();
        let query = query.to_string();
        
        let result = if Handle::try_current().is_ok() {
            tokio::task::block_in_place(|| {
                Handle::current().block_on(async {
                    let db = db.read().await;
                    db.execute_cql(&query).await
                })
            })
        } else {
            return Err("No async runtime available".to_string());
        };
        
        match result {
            Ok(coredb::QueryResult::Rows(rows)) => {
                if rows.is_empty() {
                    return Ok("<div class=\"text-gray-400\">No rows returned.</div>".to_string());
                }
                
                // Get column names from first row
                let columns: Vec<&String> = rows[0].columns.keys().collect();
                
                let mut html = String::from(
                    "<table class=\"w-full text-sm\"><thead><tr class=\"border-b border-gray-600\">"
                );
                for col in &columns {
                    html.push_str(&format!(
                        "<th class=\"px-3 py-2 text-left text-emerald-400 font-mono\">{}</th>", col
                    ));
                }
                html.push_str("</tr></thead><tbody>");
                
                for (i, row) in rows.iter().enumerate() {
                    let bg = if i % 2 == 0 { "bg-gray-800/30" } else { "" };
                    html.push_str(&format!("<tr class=\"border-b border-gray-700/50 {}\">", bg));
                    for col in &columns {
                        let val = row.columns.get(*col)
                            .map(|v| format!("{:?}", v))
                            .unwrap_or_else(|| "NULL".to_string());
                        // Truncate long values (char-safe for Unicode)
                        let display = if val.chars().count() > 100 {
                            format!("{}...", val.chars().take(100).collect::<String>())
                        } else {
                            val
                        };
                        html.push_str(&format!(
                            "<td class=\"px-3 py-2 text-gray-300 font-mono text-xs max-w-xs truncate\">{}</td>",
                            display.replace('<', "&lt;").replace('>', "&gt;")
                        ));
                    }
                    html.push_str("</tr>");
                }
                
                html.push_str("</tbody></table>");
                html.push_str(&format!(
                    "<div class=\"text-xs text-gray-500 mt-3\">{} rows returned</div>",
                    rows.len()
                ));
                
                Ok(html)
            }
            Ok(coredb::QueryResult::Success) => {
                Ok("<div class=\"text-emerald-400\">✅ Query executed successfully.</div>".to_string())
            }
            Ok(other) => {
                Ok(format!("<div class=\"text-gray-300\">{:?}</div>", other))
            }
            Err(e) => Err(format!("{}", e)),
        }
    }

    pub fn add_association(&self, _from_id: Uuid, _to_id: Uuid) -> Result<(), Box<dyn std::error::Error>> {
        // TODO: Implement associations table in CoreDB
        Ok(())
    }

    /// Get associated memories
    pub fn get_associated(&self, _id: Uuid) -> Result<Vec<MemoryItem>, Box<dyn std::error::Error>> {
        // TODO: Implement associations in CoreDB
        Ok(Vec::new())
    }

    fn parse_query_result(&self, result: coredb::QueryResult) -> Result<Vec<MemoryItem>, Box<dyn std::error::Error>> {
        let mut items = Vec::new();

        if let coredb::QueryResult::Rows(rows) = result {
            for row in rows {
                if let Some(item) = self.row_to_memory(&row) {
                    items.push(item);
                }
            }
        }

        Ok(items)
    }

    fn row_to_memory(&self, row: &coredb::query::Row) -> Option<MemoryItem> {
        let columns = &row.columns;

        let id = columns.get("id").and_then(|v| {
            if let coredb::CassandraValue::Text(s) = v {
                Uuid::parse_str(s).ok()
            } else {
                None
            }
        })?;

        let content = columns.get("content").and_then(|v| {
            if let coredb::CassandraValue::Text(s) = v {
                Some(s.clone())
            } else {
                None
            }
        })?;

        let context = columns.get("context").and_then(|v| {
            if let coredb::CassandraValue::Text(s) = v {
                if s.is_empty() { None } else { Some(s.clone()) }
            } else {
                None
            }
        });

        let memory_type = columns.get("memory_type").and_then(|v| {
            if let coredb::CassandraValue::Text(s) = v {
                match s.as_str() {
                    "Working" => Some(MemoryType::Working),
                    "Episodic" => Some(MemoryType::Episodic),
                    "Semantic" => Some(MemoryType::Semantic),
                    "Procedural" => Some(MemoryType::Procedural),
                    _ => Some(MemoryType::Semantic),
                }
            } else {
                None
            }
        }).unwrap_or(MemoryType::Semantic);

        let emotion = columns.get("emotion").and_then(|v| {
            if let coredb::CassandraValue::Text(s) = v {
                match s.as_str() {
                    "Neutral" => Some(Emotion::Neutral),
                    "Positive" => Some(Emotion::Positive),
                    "Negative" => Some(Emotion::Negative),
                    "Surprise" => Some(Emotion::Surprise),
                    _ => Some(Emotion::Neutral),
                }
            } else {
                None
            }
        }).unwrap_or(Emotion::Neutral);

        let created_at = columns.get("created_at").and_then(|v| {
            match v {
                coredb::CassandraValue::BigInt(ts) => {
                    DateTime::from_timestamp_millis(*ts).map(|dt| dt.with_timezone(&Utc))
                }
                coredb::CassandraValue::Int(ts) => {
                    DateTime::from_timestamp_millis(*ts as i64).map(|dt| dt.with_timezone(&Utc))
                }
                _ => None,
            }
        }).unwrap_or_else(Utc::now);

        let last_accessed = columns.get("last_accessed").and_then(|v| {
            match v {
                coredb::CassandraValue::BigInt(ts) => {
                    DateTime::from_timestamp_millis(*ts).map(|dt| dt.with_timezone(&Utc))
                }
                coredb::CassandraValue::Int(ts) => {
                    DateTime::from_timestamp_millis(*ts as i64).map(|dt| dt.with_timezone(&Utc))
                }
                _ => None,
            }
        }).unwrap_or_else(Utc::now);

        let access_count = columns.get("access_count").and_then(|v| {
            if let coredb::CassandraValue::Int(n) = v {
                Some(*n as u32)
            } else {
                None
            }
        }).unwrap_or(0);

        let strength = columns.get("strength").and_then(|v| {
            if let coredb::CassandraValue::Text(s) = v {
                s.parse::<f32>().ok()
            } else if let coredb::CassandraValue::Double(d) = v {
                Some(*d as f32)
            } else {
                None
            }
        }).unwrap_or(1.0);

        let embedding = columns.get("embedding").and_then(|v| {
            if let coredb::CassandraValue::Text(s) = v {
                serde_json::from_str(s).ok()
            } else {
                None
            }
        });

        let tags = columns.get("tags").and_then(|v| {
            if let coredb::CassandraValue::Text(s) = v {
                serde_json::from_str(s).ok()
            } else {
                None
            }
        }).unwrap_or_default();

        Some(MemoryItem {
            id,
            content,
            context,
            memory_type,
            emotion,
            created_at,
            last_accessed,
            access_count,
            strength,
            embedding,
            associations: Vec::new(),
            tags,
        })
    }
}
