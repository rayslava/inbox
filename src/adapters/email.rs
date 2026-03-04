use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::config::EmailConfig;
use crate::error::InboxError;
use crate::message::{IncomingMessage, MessageSource, SourceMetadata};

use super::InputAdapter;

pub struct EmailAdapter {
    pub cfg: EmailConfig,
    pub attachments_dir: std::path::PathBuf,
}

#[async_trait::async_trait]
impl InputAdapter for EmailAdapter {
    fn name(&self) -> &'static str {
        "email"
    }

    async fn run(
        self: Box<Self>,
        tx: mpsc::Sender<IncomingMessage>,
        shutdown: CancellationToken,
    ) -> Result<(), InboxError> {
        info!("Email adapter starting (IMAP IDLE)");

        loop {
            tokio::select! {
                () = shutdown.cancelled() => {
                    info!("Email adapter shutdown");
                    return Ok(());
                }
                () = run_imap_session(&self.cfg, &self.attachments_dir, &tx) => {
                    // Reconnect after session ends
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                }
            }
        }
    }
}

async fn run_imap_session(
    cfg: &EmailConfig,
    attachments_dir: &std::path::Path,
    tx: &mpsc::Sender<IncomingMessage>,
) {
    if let Err(e) = imap_idle_loop(cfg, attachments_dir, tx).await {
        warn!(?e, "IMAP session error, will retry");
    }
}

async fn imap_idle_loop(
    cfg: &EmailConfig,
    attachments_dir: &std::path::Path,
    tx: &mpsc::Sender<IncomingMessage>,
) -> Result<(), InboxError> {
    use std::sync::Arc;

    use async_imap::extensions::idle::IdleResponse;
    use rustls_pki_types::ServerName;
    use tokio_rustls::TlsConnector;
    use tokio_util::compat::TokioAsyncReadCompatExt;

    let mut root_store = rustls::RootCertStore::empty();
    root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let tls_config = rustls::ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_no_client_auth();
    let connector = TlsConnector::from(Arc::new(tls_config));

    let tcp = tokio::net::TcpStream::connect((cfg.host.as_str(), cfg.port))
        .await
        .map_err(|e| InboxError::Adapter(format!("IMAP TCP connect failed: {e}")))?;

    let server_name = ServerName::try_from(cfg.host.clone())
        .map_err(|e| InboxError::Adapter(format!("Invalid IMAP hostname: {e}")))?;
    let tls_stream = connector
        .connect(server_name, tcp)
        .await
        .map_err(|e| InboxError::Adapter(format!("IMAP TLS handshake failed: {e}")))?;

    // tokio-rustls returns a tokio stream; wrap with compat for async-imap (futures I/O).
    let client = async_imap::Client::new(tls_stream.compat());

    let mut session = client
        .login(&cfg.username, &cfg.password)
        .await
        .map_err(|(e, _)| InboxError::Adapter(format!("IMAP login failed: {e}")))?;

    session
        .select(&cfg.mailbox)
        .await
        .map_err(|e| InboxError::Adapter(format!("IMAP SELECT failed: {e}")))?;

    fetch_unseen(&mut session, cfg, attachments_dir, tx).await?;

    let mut idle = session.idle();
    idle.init()
        .await
        .map_err(|e| InboxError::Adapter(format!("IMAP IDLE init failed: {e}")))?;

    loop {
        let (idle_wait, _interrupt) = idle.wait();
        match idle_wait.await {
            Ok(IdleResponse::NewData(_) | IdleResponse::Timeout) => {
                let mut sess = idle
                    .done()
                    .await
                    .map_err(|e| InboxError::Adapter(format!("IMAP IDLE done failed: {e}")))?;
                fetch_unseen(&mut sess, cfg, attachments_dir, tx).await?;
                idle = sess.idle();
                idle.init()
                    .await
                    .map_err(|e| InboxError::Adapter(format!("IMAP IDLE re-init failed: {e}")))?;
            }
            Ok(IdleResponse::ManualInterrupt) => break,
            Err(e) => return Err(InboxError::Adapter(format!("IMAP IDLE error: {e}"))),
        }
    }

    Ok(())
}

async fn fetch_unseen<S>(
    session: &mut async_imap::Session<S>,
    _cfg: &EmailConfig,
    _attachments_dir: &std::path::Path,
    tx: &mpsc::Sender<IncomingMessage>,
) -> Result<(), InboxError>
where
    S: futures::AsyncRead + futures::AsyncWrite + Unpin + Send + std::fmt::Debug,
{
    let search = session
        .search("UNSEEN")
        .await
        .map_err(|e| InboxError::Adapter(format!("IMAP SEARCH failed: {e}")))?;

    if search.is_empty() {
        return Ok(());
    }

    let uid_set = search
        .iter()
        .map(std::string::ToString::to_string)
        .collect::<Vec<_>>()
        .join(",");

    let mut messages = session
        .fetch(&uid_set, "RFC822")
        .await
        .map_err(|e| InboxError::Adapter(format!("IMAP FETCH failed: {e}")))?;

    while let Some(msg_result) = messages.next().await {
        let msg = match msg_result {
            Ok(m) => m,
            Err(e) => {
                warn!(?e, "IMAP message fetch error");
                continue;
            }
        };

        if let Some(body) = msg.body()
            && let Some(incoming) = parse_email_raw(body)
        {
            metrics::counter!(crate::telemetry::MESSAGES_RECEIVED, "source" => "email")
                .increment(1);
            let _ = tx.send(incoming).await;
        }
    }

    Ok(())
}

/// Minimal RFC 822 parser — extracts Subject, From, and plain-text body.
fn parse_email_raw(raw: &[u8]) -> Option<IncomingMessage> {
    let raw_str = String::from_utf8_lossy(raw);
    let mut subject = String::new();
    let mut from = String::new();
    let mut message_id = None;
    let mut body_lines: Vec<&str> = Vec::new();
    let mut in_body = false;

    for line in raw_str.lines() {
        if in_body {
            body_lines.push(line);
        } else if line.is_empty() {
            in_body = true;
        } else if let Some(v) = line.strip_prefix("Subject: ") {
            v.clone_into(&mut subject);
        } else if let Some(v) = line.strip_prefix("From: ") {
            v.clone_into(&mut from);
        } else if let Some(v) = line.strip_prefix("Message-ID: ") {
            message_id = Some(v.to_owned());
        }
    }

    let text = body_lines.join("\n").trim().to_owned();

    if from.is_empty() && subject.is_empty() && text.is_empty() {
        return None;
    }

    Some(IncomingMessage::new(
        MessageSource::Email,
        if text.is_empty() {
            subject.clone()
        } else {
            text
        },
        SourceMetadata::Email {
            subject,
            from,
            message_id,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::parse_email_raw;
    use crate::message::SourceMetadata;

    #[test]
    fn parse_email_with_body_prefers_body_text() {
        let raw = b"Subject: Hi\nFrom: a@example.com\nMessage-ID: <m1>\n\nHello\nWorld";
        let msg = parse_email_raw(raw).expect("message parsed");
        assert_eq!(msg.text, "Hello\nWorld");
        match msg.metadata {
            SourceMetadata::Email {
                subject,
                from,
                message_id,
            } => {
                assert_eq!(subject, "Hi");
                assert_eq!(from, "a@example.com");
                assert_eq!(message_id.as_deref(), Some("<m1>"));
            }
            _ => panic!("expected email metadata"),
        }
    }

    #[test]
    fn parse_email_without_body_falls_back_to_subject() {
        let raw = b"Subject: SubjectOnly\nFrom: b@example.com\n\n";
        let msg = parse_email_raw(raw).expect("message parsed");
        assert_eq!(msg.text, "SubjectOnly");
    }

    #[test]
    fn parse_email_empty_returns_none() {
        let raw = b"\n\n";
        assert!(parse_email_raw(raw).is_none());
    }

    #[test]
    fn parse_email_without_headers_uses_body() {
        let raw = b"\njust body";
        let msg = parse_email_raw(raw).expect("message parsed");
        assert_eq!(msg.text, "just body");
    }
}
