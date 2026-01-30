//! CoreDB Storage Backend
//!
//! High-performance storage using CoreDB (Cassandra-style NoSQL)

#[cfg(feature = "coredb-backend")]
use coredb::{CoreDB, DatabaseConfig};

use crate::types::{MemoryItem, MemoryType, Emotion};
use chrono::{DateTime, Utc};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// CoreDB-backed storage for memory items
#[cfg(feature = "coredb-backend")]
pub struct CoreDBStorage {
    db: Arc<RwLock<CoreDB>>,
    keyspace: String,
    table: String,
    initialized: bool,
}

#[cfg(feature = "coredb-backend")]
impl CoreDBStorage {
    /// Create new CoreDB storage
    pub async fn new(data_dir: &str, keyspace: &str, table: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let config = DatabaseConfig {
            data_directory: PathBuf::from(data_dir).join("data"),
            commitlog_directory: PathBuf::from(data_dir).join("commitlog"),
            memtable_flush_threshold_mb: 16,
            compaction_throughput_mb_per_sec: 16,
            concurrent_reads: 32,
            concurrent_writes: 32,
        };

        let db = CoreDB::new(config).await?;
        
        let mut storage = Self {
            db: Arc::new(RwLock::new(db)),
            keyspace: keyspace.to_string(),
            table: table.to_string(),
            initialized: false,
        };

        storage.init_schema().await?;
        Ok(storage)
    }

    /// Initialize keyspace and table
    async fn init_schema(&mut self) -> Result<(), Box<dyn std::error::Error>> {
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

        self.initialized = true;
        Ok(())
    }

    /// Save a memory item
    pub async fn save(&self, item: &MemoryItem) -> Result<(), Box<dyn std::error::Error>> {
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
        Ok(())
    }

    /// Save multiple items in batch
    pub async fn save_batch(&self, items: &[MemoryItem]) -> Result<usize, Box<dyn std::error::Error>> {
        let mut count = 0;
        for item in items {
            if self.save(item).await.is_ok() {
                count += 1;
            }
        }
        Ok(count)
    }

    /// Search memories by content
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryItem>, Box<dyn std::error::Error>> {
        let db = self.db.read().await;

        // CoreDB doesn't support LIKE, so we fetch all and filter
        // TODO: Add full-text search index to CoreDB
        let cql = format!("SELECT * FROM {}.{}", self.keyspace, self.table);
        
        let result = db.execute_cql(&cql).await?;
        let mut items = self.parse_query_result(result)?;

        // Filter by query (simple substring match)
        let query_lower = query.to_lowercase();
        items.retain(|item| {
            item.content.to_lowercase().contains(&query_lower) ||
            item.context.as_ref().map(|c| c.to_lowercase().contains(&query_lower)).unwrap_or(false)
        });

        // Sort by strength and recency
        items.sort_by(|a, b| {
            b.strength.partial_cmp(&a.strength).unwrap_or(std::cmp::Ordering::Equal)
        });

        items.truncate(limit);
        Ok(items)
    }

    /// Get all memories
    pub async fn get_all(&self) -> Result<Vec<MemoryItem>, Box<dyn std::error::Error>> {
        let db = self.db.read().await;
        let cql = format!("SELECT * FROM {}.{}", self.keyspace, self.table);
        let result = db.execute_cql(&cql).await?;
        self.parse_query_result(result)
    }

    /// Get memory by ID
    pub async fn get(&self, id: &Uuid) -> Result<Option<MemoryItem>, Box<dyn std::error::Error>> {
        let db = self.db.read().await;
        let cql = format!(
            "SELECT * FROM {}.{} WHERE id = '{}'",
            self.keyspace, self.table, id
        );
        let result = db.execute_cql(&cql).await?;
        let items = self.parse_query_result(result)?;
        Ok(items.into_iter().next())
    }

    /// Delete a memory
    pub async fn delete(&self, id: &Uuid) -> Result<(), Box<dyn std::error::Error>> {
        // CoreDB DELETE not implemented yet, skip for now
        // let db = self.db.read().await;
        // let cql = format!("DELETE FROM {}.{} WHERE id = '{}'", self.keyspace, self.table, id);
        // db.execute_cql(&cql).await?;
        Ok(())
    }

    /// Parse CoreDB query result into MemoryItems
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

#[cfg(test)]
#[cfg(feature = "coredb-backend")]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_coredb_storage() {
        let dir = tempdir().unwrap();
        let storage = CoreDBStorage::new(
            dir.path().to_str().unwrap(),
            "test_brain",
            "memories"
        ).await.unwrap();

        // Create and save a memory
        let mut item = MemoryItem::new("Test memory content", None);
        item.tags = vec!["test".to_string()];

        storage.save(&item).await.unwrap();

        // Search for it
        let results = storage.search("Test", 10).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].content, "Test memory content");
    }
}
