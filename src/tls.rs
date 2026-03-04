use std::sync::Arc;

/// Build a `reqwest::ClientBuilder` using bundled Mozilla CA roots (webpki-roots).
///
/// # Panics
/// Panics if the default TLS protocol versions are somehow invalid (not possible in practice).
/// instead of the system certificate store.
///
/// This makes the binary self-contained for TLS and avoids startup failures on
/// systems where the system CA certificate package is not installed or where
/// `rustls-platform-verifier` cannot locate the trust store.
pub fn client_builder() -> reqwest::ClientBuilder {
    let provider = rustls::crypto::CryptoProvider::get_default()
        .cloned()
        .unwrap_or_else(|| Arc::new(rustls::crypto::aws_lc_rs::default_provider()));

    let mut root_store = rustls::RootCertStore::empty();
    root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());

    let tls_config = rustls::ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .expect("default TLS versions should always be valid")
        .with_root_certificates(root_store)
        .with_no_client_auth();

    reqwest::Client::builder().use_preconfigured_tls(tls_config)
}
