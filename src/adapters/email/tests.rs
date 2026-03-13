use tokio::sync::mpsc;

use super::{imap_idle_loop, parse_email_raw};
use crate::config::EmailConfig;
use crate::message::SourceMetadata;

// ── parse_email_raw unit tests ────────────────────────────────────────────────

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

// ── IMAP integration mock tests ───────────────────────────────────────────────

fn test_email_cfg(host: &str, port: u16) -> EmailConfig {
    EmailConfig {
        enabled: true,
        host: host.to_owned(),
        port,
        username: "user".to_owned(),
        password: "pass".to_owned(),
        mailbox: "INBOX".to_owned(),
        mark_as_seen: false,
        processed_mailbox: String::new(),
        tls: false,
    }
}

/// Spawn a one-shot mock IMAP server on a random port.
async fn spawn_mock_imap<F, Fut>(response_fn: F) -> u16
where
    F: FnOnce(tokio::net::TcpStream) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = ()> + Send,
{
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let port = listener.local_addr().expect("local_addr").port();
    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.expect("accept");
        response_fn(stream).await;
    });
    port
}

/// Write a line to the IMAP stream.
async fn imap_write(stream: &mut tokio::net::TcpStream, line: &str) {
    use tokio::io::AsyncWriteExt;
    stream.write_all(line.as_bytes()).await.expect("write");
}

/// Read one CRLF-terminated line from the stream.
/// Returns the tag (first token) and the full line.
async fn imap_read_cmd(stream: &mut tokio::net::TcpStream) -> (String, String) {
    use tokio::io::AsyncReadExt;
    let mut line = Vec::with_capacity(128);
    let mut byte = [0u8; 1];
    loop {
        let n = stream.read(&mut byte).await.unwrap_or(0);
        if n == 0 {
            break;
        }
        line.push(byte[0]);
        if line.ends_with(b"\r\n") {
            break;
        }
    }
    let s = String::from_utf8_lossy(&line).into_owned();
    let tag = s.split_whitespace().next().unwrap_or("TAG").to_owned();
    (tag, s)
}

#[tokio::test]
async fn imap_idle_loop_login_failure_returns_error() {
    let port = spawn_mock_imap(|mut stream| async move {
        imap_write(&mut stream, "* OK IMAP ready\r\n").await;
        let (tag, _) = imap_read_cmd(&mut stream).await; // LOGIN
        imap_write(&mut stream, &format!("{tag} NO LOGIN failed\r\n")).await;
        // Stream drops here — client sees EOF
    })
    .await;

    let cfg = test_email_cfg("127.0.0.1", port);
    let (tx, _rx) = mpsc::channel(10);
    let result = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        imap_idle_loop(&cfg, std::path::Path::new("/tmp"), &tx).await
    })
    .await
    .expect("test timed out");
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string().to_ascii_lowercase();
    assert!(
        msg.contains("login") || msg.contains("imap"),
        "unexpected error: {msg}"
    );
}

#[tokio::test]
async fn imap_idle_loop_select_failure_returns_error() {
    let port = spawn_mock_imap(|mut stream| async move {
        imap_write(&mut stream, "* OK IMAP ready\r\n").await;
        let (tag, _) = imap_read_cmd(&mut stream).await; // LOGIN
        imap_write(&mut stream, &format!("{tag} OK logged in\r\n")).await;
        let (tag2, _) = imap_read_cmd(&mut stream).await; // SELECT
        imap_write(&mut stream, &format!("{tag2} NO SELECT failed\r\n")).await;
    })
    .await;

    let cfg = test_email_cfg("127.0.0.1", port);
    let (tx, _rx) = mpsc::channel(10);
    let result = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        imap_idle_loop(&cfg, std::path::Path::new("/tmp"), &tx).await
    })
    .await
    .expect("test timed out");
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string().to_ascii_lowercase();
    assert!(
        msg.contains("select") || msg.contains("imap"),
        "unexpected error: {msg}"
    );
}

#[tokio::test]
async fn imap_idle_loop_no_unseen_messages_channel_empty() {
    let port = spawn_mock_imap(|mut stream| async move {
        imap_write(&mut stream, "* OK IMAP ready\r\n").await;
        let (tag, _) = imap_read_cmd(&mut stream).await; // LOGIN
        imap_write(&mut stream, &format!("{tag} OK logged in\r\n")).await;
        let (tag2, _) = imap_read_cmd(&mut stream).await; // SELECT
        imap_write(&mut stream, "* 0 EXISTS\r\n").await;
        imap_write(
            &mut stream,
            &format!("{tag2} OK [READ-WRITE] SELECT complete\r\n"),
        )
        .await;
        let (tag3, _) = imap_read_cmd(&mut stream).await; // SEARCH UNSEEN
        imap_write(&mut stream, "* SEARCH\r\n").await;
        imap_write(&mut stream, &format!("{tag3} OK SEARCH complete\r\n")).await;
        // Drop stream — IDLE init will fail and imap_idle_loop returns Err,
        // but fetch_unseen already completed with no messages.
    })
    .await;

    let cfg = test_email_cfg("127.0.0.1", port);
    let (tx, mut rx) = mpsc::channel(10);
    // We don't care about Ok/Err here; only that the channel stays empty.
    let _ = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        imap_idle_loop(&cfg, std::path::Path::new("/tmp"), &tx).await
    })
    .await;
    assert!(rx.try_recv().is_err(), "expected empty channel");
}

#[tokio::test]
async fn imap_idle_loop_fetches_message_onto_channel() {
    let email_raw = "From: sender@example.com\r\nSubject: Hello\r\n\r\nBody text\r\n";
    let email_len = email_raw.len();

    let port = spawn_mock_imap(move |mut stream| async move {
        imap_write(&mut stream, "* OK IMAP ready\r\n").await;
        let (tag, _) = imap_read_cmd(&mut stream).await; // LOGIN
        imap_write(&mut stream, &format!("{tag} OK logged in\r\n")).await;
        let (tag2, _) = imap_read_cmd(&mut stream).await; // SELECT
        imap_write(&mut stream, "* 1 EXISTS\r\n").await;
        imap_write(
            &mut stream,
            &format!("{tag2} OK [READ-WRITE] SELECT complete\r\n"),
        )
        .await;
        let (tag3, _) = imap_read_cmd(&mut stream).await; // SEARCH UNSEEN
        imap_write(&mut stream, "* SEARCH 1\r\n").await;
        imap_write(&mut stream, &format!("{tag3} OK SEARCH complete\r\n")).await;
        let (tag4, _) = imap_read_cmd(&mut stream).await; // FETCH 1 RFC822
        imap_write(
            &mut stream,
            &format!("* 1 FETCH (RFC822 {{{email_len}}}\r\n"),
        )
        .await;
        imap_write(&mut stream, email_raw).await;
        imap_write(&mut stream, ")\r\n").await;
        imap_write(&mut stream, &format!("{tag4} OK FETCH complete\r\n")).await;
        // Drop stream — IDLE init fails and imap_idle_loop returns Err,
        // but the message is already on the channel.
    })
    .await;

    let cfg = test_email_cfg("127.0.0.1", port);
    let (tx, mut rx) = mpsc::channel(10);
    let _ = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        imap_idle_loop(&cfg, std::path::Path::new("/tmp"), &tx).await
    })
    .await;

    let received = rx.try_recv().expect("expected one message on channel");
    assert_eq!(received.text, "Body text");
    match received.metadata {
        SourceMetadata::Email { subject, from, .. } => {
            assert_eq!(subject, "Hello");
            assert_eq!(from, "sender@example.com");
        }
        _ => panic!("expected email metadata"),
    }
}
