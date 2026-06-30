use anodized::spec;
use async_trait::async_trait;
use inbox_core::{CoreError, EmbeddingProvider};

use crate::error::InboxError;

pub struct EmbedClient {
    endpoint: String,
    model: String,
    api_key: Option<String>,
    client: reqwest::Client,
}

impl EmbedClient {
    /// # Errors
    /// Returns an error if the HTTP client cannot be built.
    pub fn new(
        endpoint: String,
        model: String,
        api_key: Option<String>,
    ) -> Result<Self, InboxError> {
        let client = crate::tls::client_builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| InboxError::Memory(format!("Failed to build embed HTTP client: {e}")))?;
        Ok(Self {
            endpoint,
            model,
            api_key,
            client,
        })
    }

    /// Embed `text` and return the embedding vector.
    ///
    /// # Errors
    /// Returns an error if the HTTP request fails or the response is unparseable.
    #[spec(requires: !text.is_empty())]
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, InboxError> {
        // Uses Ollama's native POST /api/embed endpoint.
        // Response: {"embeddings": [[...f32 vector...]]}
        let url = format!("{}/api/embed", self.endpoint);
        let body = serde_json::json!({
            "input": text,
            "model": self.model,
        });

        let mut req = self.client.post(&url).json(&body);
        if let Some(key) = &self.api_key
            && !key.is_empty()
        {
            req = req.bearer_auth(key);
        }

        let resp = req
            .send()
            .await
            .map_err(|e| InboxError::Memory(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(InboxError::Memory(format!(
                "Embedding API error {status}: {body}"
            )));
        }

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| InboxError::Memory(format!("Embedding parse error: {e}")))?;

        let embedding: Vec<f32> = json["embeddings"][0]
            .as_array()
            .ok_or_else(|| InboxError::Memory("Missing embeddings[0] in response".into()))?
            .iter()
            .filter_map(|v| serde_json::from_value::<f32>(v.clone()).ok())
            .collect();

        if embedding.is_empty() {
            return Err(InboxError::Memory("Empty embedding vector".into()));
        }

        Ok(embedding)
    }
}

/// Bridges the concrete Ollama-native client to the `inbox_core` trait boundary,
/// mapping `InboxError` into the dependency-light `CoreError`.
#[async_trait]
impl EmbeddingProvider for EmbedClient {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, CoreError> {
        EmbedClient::embed(self, text)
            .await
            .map_err(CoreError::from)
    }
}
