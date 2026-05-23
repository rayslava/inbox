use std::sync::Arc;

/// Build a `reqwest::ClientBuilder` using bundled Mozilla CA roots (webpki-roots)
/// instead of the system certificate store.
///
/// This makes the binary self-contained for TLS and avoids startup failures on
/// systems where the system CA certificate package is not installed or where
/// `rustls-platform-verifier` cannot locate the trust store.
///
/// If the preconfigured rustls protocol-version setup fails (not possible in
/// practice), this degrades to reqwest's built-in TLS rather than panicking.
pub fn client_builder() -> reqwest::ClientBuilder {
    let provider = rustls::crypto::CryptoProvider::get_default()
        .cloned()
        .unwrap_or_else(|| Arc::new(rustls::crypto::aws_lc_rs::default_provider()));

    let mut root_store = rustls::RootCertStore::empty();
    root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());

    match rustls::ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
    {
        Ok(cfg) => {
            let tls_config = cfg.with_root_certificates(root_store).with_no_client_auth();
            reqwest::Client::builder().use_preconfigured_tls(tls_config)
        }
        Err(e) => {
            tracing::error!(
                ?e,
                "TLS protocol-version setup failed; using reqwest default TLS"
            );
            reqwest::Client::builder()
        }
    }
}
