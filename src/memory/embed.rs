use anodized::spec;
use async_trait::async_trait;
use inbox_core::{CoreError, EmbeddingProvider};

use crate::config::EmbeddingApi;
use crate::error::InboxError;

pub struct EmbedClient {
    endpoint: String,
    api: EmbeddingApi,
    model: String,
    api_key: Option<String>,
    client: reqwest::Client,
}

impl EmbedClient {
    /// # Errors
    /// Returns an error if the HTTP client cannot be built.
    pub fn new(
        endpoint: String,
        api: EmbeddingApi,
        model: String,
        api_key: Option<String>,
    ) -> Result<Self, InboxError> {
        let client = crate::tls::client_builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| InboxError::Memory(format!("Failed to build embed HTTP client: {e}")))?;
        Ok(Self {
            endpoint,
            api,
            model,
            api_key,
            client,
        })
    }

    /// Embed `text` and return the embedding vector.
    ///
    /// Routes to the Ollama-native `/api/embed` or the OpenAI-compatible
    /// `/embeddings` endpoint per the configured [`EmbeddingApi`].
    ///
    /// # Errors
    /// Returns an error if the HTTP request fails or the response is unparseable.
    #[spec(requires: !text.is_empty())]
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, InboxError> {
        let path = match self.api {
            EmbeddingApi::Ollama => "/api/embed",
            EmbeddingApi::Openai => "/embeddings",
        };
        let url = format!("{}{path}", self.endpoint);
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

        // Ollama nests the vector under `embeddings[0]`; OpenAI under
        // `data[0].embedding`.
        let vector = match self.api {
            EmbeddingApi::Ollama => &json["embeddings"][0],
            EmbeddingApi::Openai => &json["data"][0]["embedding"],
        };
        // Parse strictly: a non-numeric element (e.g. `null`) must fail, not be
        // silently dropped — a truncated vector would corrupt dimension
        // detection and the vector index.
        let embedding: Vec<f32> = vector
            .as_array()
            .ok_or_else(|| InboxError::Memory("Missing embedding vector in response".into()))?
            .iter()
            .map(|v| serde_json::from_value::<f32>(v.clone()))
            .collect::<Result<_, _>>()
            .map_err(|e| InboxError::Memory(format!("Malformed embedding element: {e}")))?;

        if embedding.is_empty() {
            return Err(InboxError::Memory("Empty embedding vector".into()));
        }

        Ok(embedding)
    }
}

/// Bridges the concrete embedding client to the `inbox_core` trait boundary,
/// mapping `InboxError` into the dependency-light `CoreError`.
#[async_trait]
impl EmbeddingProvider for EmbedClient {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, CoreError> {
        EmbedClient::embed(self, text)
            .await
            .map_err(CoreError::from)
    }
}
