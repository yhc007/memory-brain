//! Storage - SQLite backend for persistent memory

use crate::types::MemoryItem;
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

pub struct Storage {
    conn: Arc<Mutex<Connection>>,
    table_name: String,
}

impl Storage {
    pub fn new(db_path: &str, table_name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let conn = Connection::open(db_path)?;
        let storage = Self {
            conn: Arc::new(Mutex::new(conn)),
            table_name: table_name.to_string(),
        };
        storage.init_tables()?;
        Ok(storage)
    }

    fn init_tables(&self) -> Result<(), Box<dyn std::error::Error>> {
        let conn = self.conn.lock().unwrap();
        
        // Main memory table
        conn.execute(
            &format!(
                "CREATE TABLE IF NOT EXISTS {} (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    context TEXT,
                    memory_type TEXT NOT NULL,
                    emotion TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    access_count INTEGER NOT NULL,
                    strength REAL NOT NULL,
                    embedding TEXT,
                    tags TEXT
                )",
                self.table_name
            ),
            [],
        )?;

        // Associations table
        conn.execute(
            &format!(
                "CREATE TABLE IF NOT EXISTS {}_associations (
                    from_id TEXT NOT NULL,
                    to_id TEXT NOT NULL,
                    strength REAL DEFAULT 1.0,
                    PRIMARY KEY (from_id, to_id)
                )",
                self.table_name
            ),
            [],
        )?;

        // Index for faster searching
        conn.execute(
            &format!(
                "CREATE INDEX IF NOT EXISTS idx_{}_content ON {} (content)",
                self.table_name, self.table_name
            ),
            [],
        )?;

        Ok(())
    }

    /// Save a memory item
    pub fn save(&self, item: &MemoryItem) -> Result<(), Box<dyn std::error::Error>> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            &format!(
                "INSERT OR REPLACE INTO {} 
                (id, content, context, memory_type, emotion, created_at, last_accessed, access_count, strength, embedding, tags)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                self.table_name
            ),
            params![
                item.id.to_string(),
                item.content,
                item.context,
                serde_json::to_string(&item.memory_type)?,
                serde_json::to_string(&item.emotion)?,
                item.created_at.to_rfc3339(),
                item.last_accessed.to_rfc3339(),
                item.access_count,
                item.strength,
                item.embedding.as_ref().map(|e| serde_json::to_string(e).ok()).flatten(),
                serde_json::to_string(&item.tags)?,
            ],
        )?;
        Ok(())
    }

    /// Update a memory item
    pub fn update(&self, item: &MemoryItem) -> Result<(), Box<dyn std::error::Error>> {
        self.save(item)
    }

    /// Delete a memory item
    pub fn delete(&self, id: &Uuid) -> Result<(), Box<dyn std::error::Error>> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            &format!("DELETE FROM {} WHERE id = ?1", self.table_name),
            params![id.to_string()],
        )?;
        // Also delete associations
        conn.execute(
            &format!("DELETE FROM {}_associations WHERE from_id = ?1 OR to_id = ?1", self.table_name),
            params![id.to_string()],
        )?;
        Ok(())
    }

    /// Search memories by content (simple text search)
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryItem>, Box<dyn std::error::Error>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(&format!(
            "SELECT id, content, context, memory_type, emotion, created_at, last_accessed, access_count, strength, embedding, tags 
             FROM {} 
             WHERE content LIKE ?1 OR context LIKE ?1
             ORDER BY strength DESC, last_accessed DESC
             LIMIT ?2",
            self.table_name
        ))?;

        let pattern = format!("%{}%", query);
        let items = stmt.query_map(params![pattern, limit as i64], |row| {
            Ok(self.row_to_memory(row))
        })?
        .filter_map(|r| r.ok())
        .filter_map(|r| r.ok())
        .collect();

        Ok(items)
    }

    /// Get all memories
    pub fn get_all(&self) -> Result<Vec<MemoryItem>, Box<dyn std::error::Error>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(&format!(
            "SELECT id, content, context, memory_type, emotion, created_at, last_accessed, access_count, strength, embedding, tags 
             FROM {}",
            self.table_name
        ))?;

        let items = stmt.query_map([], |row| {
            Ok(self.row_to_memory(row))
        })?
        .filter_map(|r| r.ok())
        .filter_map(|r| r.ok())
        .collect();

        Ok(items)
    }

    /// Get recent memories
    pub fn get_recent(&self, limit: usize) -> Result<Vec<MemoryItem>, Box<dyn std::error::Error>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(&format!(
            "SELECT id, content, context, memory_type, emotion, created_at, last_accessed, access_count, strength, embedding, tags 
             FROM {} 
             ORDER BY created_at DESC
             LIMIT ?1",
            self.table_name
        ))?;

        let items = stmt.query_map(params![limit as i64], |row| {
            Ok(self.row_to_memory(row))
        })?
        .filter_map(|r| r.ok())
        .filter_map(|r| r.ok())
        .collect();

        Ok(items)
    }

    /// Get memories by time range
    pub fn get_by_time_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<MemoryItem>, Box<dyn std::error::Error>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(&format!(
            "SELECT id, content, context, memory_type, emotion, created_at, last_accessed, access_count, strength, embedding, tags 
             FROM {} 
             WHERE created_at >= ?1 AND created_at <= ?2
             ORDER BY created_at DESC",
            self.table_name
        ))?;

        let items = stmt.query_map(params![start.to_rfc3339(), end.to_rfc3339()], |row| {
            Ok(self.row_to_memory(row))
        })?
        .filter_map(|r| r.ok())
        .filter_map(|r| r.ok())
        .collect();

        Ok(items)
    }

    /// Get memories by tag
    pub fn get_by_tag(&self, tag: &str) -> Result<Vec<MemoryItem>, Box<dyn std::error::Error>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(&format!(
            "SELECT id, content, context, memory_type, emotion, created_at, last_accessed, access_count, strength, embedding, tags 
             FROM {} 
             WHERE tags LIKE ?1
             ORDER BY strength DESC",
            self.table_name
        ))?;

        let pattern = format!("%\"{}%", tag);
        let items = stmt.query_map(params![pattern], |row| {
            Ok(self.row_to_memory(row))
        })?
        .filter_map(|r| r.ok())
        .filter_map(|r| r.ok())
        .collect();

        Ok(items)
    }

    /// Add association between memories
    pub fn add_association(&self, from_id: Uuid, to_id: Uuid) -> Result<(), Box<dyn std::error::Error>> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            &format!(
                "INSERT OR IGNORE INTO {}_associations (from_id, to_id) VALUES (?1, ?2)",
                self.table_name
            ),
            params![from_id.to_string(), to_id.to_string()],
        )?;
        Ok(())
    }

    /// Get associated memories
    pub fn get_associated(&self, id: Uuid) -> Result<Vec<MemoryItem>, Box<dyn std::error::Error>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(&format!(
            "SELECT m.id, m.content, m.context, m.memory_type, m.emotion, m.created_at, m.last_accessed, m.access_count, m.strength, m.embedding, m.tags 
             FROM {} m
             JOIN {}_associations a ON m.id = a.to_id
             WHERE a.from_id = ?1",
            self.table_name, self.table_name
        ))?;

        let items = stmt.query_map(params![id.to_string()], |row| {
            Ok(self.row_to_memory(row))
        })?
        .filter_map(|r| r.ok())
        .filter_map(|r| r.ok())
        .collect();

        Ok(items)
    }

    fn row_to_memory(&self, row: &rusqlite::Row) -> Result<MemoryItem, rusqlite::Error> {
        let id_str: String = row.get(0)?;
        let content: String = row.get(1)?;
        let context: Option<String> = row.get(2)?;
        let memory_type_str: String = row.get(3)?;
        let emotion_str: String = row.get(4)?;
        let created_at_str: String = row.get(5)?;
        let last_accessed_str: String = row.get(6)?;
        let access_count: u32 = row.get(7)?;
        let strength: f32 = row.get(8)?;
        let embedding_str: Option<String> = row.get(9)?;
        let tags_str: String = row.get(10)?;

        Ok(MemoryItem {
            id: Uuid::parse_str(&id_str).unwrap_or_else(|_| Uuid::new_v4()),
            content,
            context,
            memory_type: serde_json::from_str(&memory_type_str).unwrap_or(crate::types::MemoryType::Working),
            emotion: serde_json::from_str(&emotion_str).unwrap_or(crate::types::Emotion::Neutral),
            created_at: DateTime::parse_from_rfc3339(&created_at_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            last_accessed: DateTime::parse_from_rfc3339(&last_accessed_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            access_count,
            strength,
            embedding: embedding_str.and_then(|s| serde_json::from_str(&s).ok()),
            associations: Vec::new(), // Loaded separately if needed
            tags: serde_json::from_str(&tags_str).unwrap_or_default(),
        })
    }
}
