use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use color_eyre::eyre::{Context, Result};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use inbox::{
    adapters::{InputAdapter, email::EmailAdapter, http::HttpAdapter, telegram::TelegramAdapter},
    config,
    health::ReadinessState,
    llm, log_capture,
    output::{OutputWriter, org_file::OrgFileWriter},
    pending::PendingStore,
    pipeline::Pipeline,
    processing_status::ProcessingTracker,
    resume_task::{self, ResumeTaskArgs},
    telemetry as inbox_telemetry, web,
};

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "inbox", about = "Universal inbox daemon")]
struct Cli {
    #[arg(short, long, default_value = "config.toml")]
    config: PathBuf,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate an Argon2id password hash for use in `admin.password_hash`
    HashPassword,
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;
    let cli = Cli::parse();

    if let Some(Commands::HashPassword) = cli.command {
        return hash_password_cmd();
    }

    // Load config
    let cfg = config::load(&cli.config)
        .with_context(|| format!("Failed to load config from {}", cli.config.display()))?;

    // Logging
    let log_store = log_capture::LogStore::new(log_capture::CAPACITY);
    init_logging(
        &cfg.general.log_format,
        &cfg.general.log_level,
        Arc::clone(&log_store),
    );

    info!(version = env!("CARGO_PKG_VERSION"), "inbox starting");

    // Metrics
    let prometheus_handle = metrics_exporter_prometheus::PrometheusBuilder::new()
        .build_recorder()
        .handle();
    inbox_telemetry::describe_metrics();

    // Shared state
    let readiness = ReadinessState::new(false);
    let session_store = web::auth::new_session_store();
    let shutdown = CancellationToken::new();
    let cfg = Arc::new(cfg);

    // Build pipeline
    let llm::BuildResult {
        chain: llm_chain,
        memory_store,
    } = llm::build_chain(&cfg);
    let llm_chain = Arc::new(llm_chain);
    let writer = Arc::new(OrgFileWriter) as Arc<dyn OutputWriter>;
    let tracker = Arc::new(ProcessingTracker::new());

    // Open the pending store if resume is enabled.
    let pending_store: Option<Arc<PendingStore>> = if cfg.pipeline.resume.enabled {
        let db_path = cfg
            .pipeline
            .resume
            .db_path
            .clone()
            .unwrap_or_else(|| cfg.general.attachments_dir.join("pending.db"));
        match PendingStore::open(&db_path).await {
            Ok(s) => {
                info!(?db_path, "Pending store opened");
                Some(Arc::new(s))
            }
            Err(e) => {
                warn!(?e, "Failed to open pending store — resume disabled");
                None
            }
        }
    } else {
        None
    };

    let pipeline = Arc::new(Pipeline::new(
        Arc::clone(&cfg),
        llm_chain,
        writer,
        Arc::clone(&tracker),
        memory_store.clone(),
        pending_store.clone(),
    ));

    let (tx, rx) = mpsc::channel::<inbox::message::IncomingMessage>(256);

    // Spawn pipeline consumer
    let pipeline_clone = Arc::clone(&pipeline);
    tokio::spawn(async move { pipeline_clone.run(rx).await });

    // Spawn background resume task if the pending store is available.
    if let Some(store) = pending_store {
        let args = ResumeTaskArgs {
            store,
            pipeline: Arc::clone(&pipeline),
            config: Arc::clone(&cfg),
            telegram_notifier: None, // TODO: wire Telegram bot when adapter exposes it
            shutdown: shutdown.clone(),
        };
        tokio::spawn(resume_task::run(args));
        info!("Background resume task started");
    }

    // Spawn adapters
    spawn_adapters(&cfg, &tx, &shutdown, memory_store.as_ref());

    // Admin server
    {
        let admin_addr = cfg.admin.bind_addr;
        let admin_router = web::admin_router(web::AdminRouterArgs {
            cfg: Arc::clone(&cfg),
            readiness: readiness.clone(),
            session_store,
            metrics_handle: prometheus_handle,
            log_store: Arc::clone(&log_store),
            tracker: Arc::clone(&tracker),
            inbox_tx: Some(tx.clone()),
            attachments_dir: cfg.general.attachments_dir.clone(),
            memory_store: memory_store.clone(),
        });
        tokio::spawn(async move {
            let listener = tokio::net::TcpListener::bind(admin_addr)
                .await
                .expect("Failed to bind admin server");
            info!(%admin_addr, "Admin server listening");
            axum::serve(listener, admin_router)
                .await
                .expect("Admin server error");
        });
    }

    // Mark ready
    readiness.set_ready();
    info!("Inbox ready");

    // Wait for shutdown signal
    wait_for_shutdown_signal().await;

    // Graceful shutdown
    info!("Shutdown signal received");
    readiness.set_not_ready();

    let drain_secs = cfg.admin.shutdown_drain_secs;
    if drain_secs > 0 {
        info!(drain_secs, "Draining (waiting for load balancer)");
        tokio::time::sleep(std::time::Duration::from_secs(drain_secs)).await;
    }

    shutdown.cancel();
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    info!("Inbox exiting");
    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn init_logging(format: &str, level: &str, log_store: std::sync::Arc<log_capture::LogStore>) {
    use tracing_subscriber::{EnvFilter, fmt, prelude::*};

    let filter = EnvFilter::try_from_env("RUST_LOG")
        .or_else(|_| EnvFilter::try_new(format!("inbox={level}")))
        .unwrap_or_else(|_| EnvFilter::new("info"));

    let capture = log_capture::LogCaptureLayer::new(log_store);
    let registry = tracing_subscriber::registry().with(filter).with(capture);

    let log_format = std::env::var("LOG_FORMAT").unwrap_or_else(|_| format.to_owned());

    if log_format == "json" {
        registry.with(fmt::layer().json()).init();
    } else {
        registry.with(fmt::layer().pretty()).init();
    }
}

fn spawn_adapters(
    cfg: &Arc<config::Config>,
    tx: &mpsc::Sender<inbox::message::IncomingMessage>,
    shutdown: &CancellationToken,
    memory_store: Option<&std::sync::Arc<inbox::memory::MemoryStore>>,
) {
    if cfg.adapters.http.enabled {
        let adapter = Box::new(HttpAdapter {
            cfg: cfg.adapters.http.clone(),
            attachments_dir: cfg.general.attachments_dir.clone(),
        });
        let tx2 = tx.clone();
        let sd = shutdown.clone();
        tokio::spawn(async move {
            if let Err(e) = adapter.run(tx2, sd).await {
                warn!(?e, "HTTP adapter exited with error");
            }
        });
    }

    if cfg.adapters.telegram.enabled {
        let adapter = Box::new(TelegramAdapter {
            cfg: cfg.adapters.telegram.clone(),
            attachments_dir: cfg.general.attachments_dir.clone(),
            memory_store: memory_store.cloned(),
        });
        let tx2 = tx.clone();
        let sd = shutdown.clone();
        tokio::spawn(async move {
            if let Err(e) = adapter.run(tx2, sd).await {
                warn!(?e, "Telegram adapter exited with error");
            }
        });
    }

    if cfg.adapters.email.enabled {
        let adapter = Box::new(EmailAdapter {
            cfg: cfg.adapters.email.clone(),
            attachments_dir: cfg.general.attachments_dir.clone(),
        });
        let tx2 = tx.clone();
        let sd = shutdown.clone();
        tokio::spawn(async move {
            if let Err(e) = adapter.run(tx2, sd).await {
                warn!(?e, "Email adapter exited with error");
            }
        });
    }
}

async fn wait_for_shutdown_signal() {
    use tokio::signal;

    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut sigterm = signal(SignalKind::terminate()).expect("SIGTERM handler");
        tokio::select! {
            _ = signal::ctrl_c() => {},
            _ = sigterm.recv() => {},
        }
    }
    #[cfg(not(unix))]
    {
        let _ = signal::ctrl_c().await;
    }
}

fn hash_password_cmd() -> Result<()> {
    use argon2::{Argon2, PasswordHasher};

    // Read password from stdin (for scripting) or interactively
    let mut password = String::new();
    std::io::stdin().read_line(&mut password).ok();
    let password = password.trim();

    // argon2 0.6+ with `getrandom` feature generates salt automatically
    let hash = Argon2::default()
        .hash_password(password.as_bytes())
        .map_err(|e| color_eyre::eyre::eyre!("Hash error: {e}"))?
        .to_string();

    println!("{hash}");
    Ok(())
}
