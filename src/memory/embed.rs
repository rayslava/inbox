use anodized::spec;

use crate::error::InboxError;

pub struct EmbedClient {
    endpoint: String,
    model: String,
    api_key: Option<String>,
    client: reqwest::Client,
}

impl EmbedClient {
    /// # Panics
    /// Panics if the TLS backend cannot be initialised (extremely unlikely in practice).
    #[must_use]
    pub fn new(endpoint: String, model: String, api_key: Option<String>) -> Self {
        let client = crate::tls::client_builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("Failed to build embed HTTP client");
        Self {
            endpoint,
            model,
            api_key,
            client,
        }
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
        if let Some(key) = &self.api_key {
            if !key.is_empty() {
                req = req.bearer_auth(key);
            }
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
