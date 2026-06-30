//! Bridges the concrete grafeo-backed `MemoryStore` to the `inbox_core`
//! `VectorStore` trait, delegating to the inherent methods and mapping
//! `InboxError` into the dependency-light `CoreError`.

use async_trait::async_trait;
use inbox_core::{CoreError, MemoryEntry, SourceEntry, VectorStore};

use super::MemoryStore;

#[async_trait]
impl VectorStore for MemoryStore {
    async fn save(&self, key: &str, value: &str) -> Result<(), CoreError> {
        MemoryStore::save(self, key, value)
            .await
            .map_err(CoreError::from)
    }

    async fn link_source(
        &self,
        memory_key: &str,
        source_kind: &str,
        source_id: &str,
        title: &str,
    ) -> Result<(), CoreError> {
        MemoryStore::link_source(self, memory_key, source_kind, source_id, title)
            .await
            .map_err(CoreError::from)
    }

    async fn link_memories(
        &self,
        from_key: &str,
        to_key: &str,
        relation: &str,
    ) -> Result<(), CoreError> {
        MemoryStore::link_memories(self, from_key, to_key, relation)
            .await
            .map_err(CoreError::from)
    }

    async fn recall(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>, CoreError> {
        MemoryStore::recall(self, query, limit)
            .await
            .map_err(CoreError::from)
    }

    async fn context(&self, query: &str, hops: u32) -> Result<Vec<MemoryEntry>, CoreError> {
        MemoryStore::context(self, query, hops)
            .await
            .map_err(CoreError::from)
    }

    async fn sources(&self, memory_key: &str) -> Result<Vec<SourceEntry>, CoreError> {
        MemoryStore::sources(self, memory_key)
            .await
            .map_err(CoreError::from)
    }
}
