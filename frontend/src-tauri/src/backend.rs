// src-tauri/src/backend.rs
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::path::PathBuf;
use std::io::{BufRead, BufReader};
use std::thread;
use tauri::{AppHandle, Manager};

#[cfg(windows)]
use std::os::windows::io::AsRawHandle;

#[derive(Debug)]
pub struct BackendProcess {
    process: Option<Child>,
    port: u16,
    // Ties the Python child's lifetime to this process at the OS level, so
    // it dies even if app.exe is force-killed (taskkill /F, a crash, Task
    // Manager "End task") and no Rust cleanup code -- Drop, the tray "Quit"
    // handler, none of it -- ever gets to run. See start()'s comment below
    // for why a Windows Job Object is the only thing that can guarantee
    // this; `Option` because job-object setup failure shouldn't fail the
    // whole backend start, just lose this specific safety net.
    #[cfg(windows)]
    job: Option<win32job::Job>,
}

impl BackendProcess {
    pub fn new(port: u16) -> Self {
        Self {
            process: None,
            port,
            #[cfg(windows)]
            job: None,
        }
    }

    pub fn start(&mut self, runtime: &PythonRuntime) -> Result<(), String> {
        if self.is_running() {
            return Ok(()); // Already running
        }

        log::info!("Starting Python backend from: {:?}", runtime.working_dir);
        log::info!("Using Python: {:?}", runtime.python_cmd);
        log::info!("Backend will run on port: {}", self.port);

        // Start Python backend with backend/app.py (FastAPI server)
        // Pipe output so we can log it
        let mut child = Command::new(&runtime.python_cmd)
            .args(["-m", "backend.app"])  // Run as module: python -m backend.app
            .current_dir(&runtime.working_dir)
            .env("ORION_DATA_DIR", &runtime.data_dir)
            .env("ORION_VECTORSTORE_PERSIST_DIRECTORY", runtime.data_dir.join("chroma-data"))
            .env("PYTHONUNBUFFERED", "1")
            // On Windows, Python only gets real UTF-8 stdio automatically when
            // attached to an interactive console -- piped here (this process
            // captures stdout/stderr below), it falls back to the legacy system
            // codepage (cp1252), which can't encode the emoji/checkmarks used
            // throughout the codebase's logging. That crashed in multiple
            // independent places (the `logging` module, `rich.Console`, plain
            // `print()`/`open()`) because each one decides its own encoding --
            // no single in-code fix covers all of them. PYTHONUTF8=1 (PEP 540)
            // sets Python's default I/O encoding globally before any of that
            // code runs, closing the whole bug class in one place instead of
            // chasing it writer-by-writer. See CLAUDE.md / Eru Polishing.md for
            // the charmap bug history this replaces piecemeal fixes for.
            .env("PYTHONUTF8", "1")
            // Tells backend/app.py whether it's safe to skip Uvicorn's
            // reload=True file-watcher supervisor -- only relevant when
            // launched via `python -m backend.app` (dev's `uvicorn --reload`
            // invocation doesn't go through this env var at all).
            .env("ORION_PACKAGED", if runtime.is_packaged { "1" } else { "0" })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!(
                "Failed to start Python backend: {}. Tried to run {:?} -m backend.app from {:?}",
                e, runtime.python_cmd, runtime.working_dir
            ))?;

        log::info!("Python backend process started with PID: {:?}", child.id());

        // Assign the child to a Windows Job Object with kill-on-job-close,
        // so it dies automatically if this process is ever force-killed
        // (taskkill /F, a crash, Task Manager "End task") rather than being
        // orphaned. Windows has no built-in parent/child process-tree
        // relationship -- Child::kill() in stop()/Drop below only reaches
        // this one PID, and force-killing app.exe runs zero Rust code (no
        // Drop, no exit handler), so without this the Python process (and
        // in dev mode, reload=True's own worker subchild -- processes join
        // their parent's job by default) would simply survive. This is a
        // pure safety net alongside the existing explicit kill() path, not
        // a replacement for it: a failure here is logged and otherwise
        // ignored, since the backend runs fine without it, just loses this
        // specific guarantee.
        #[cfg(windows)]
        {
            let mut info = win32job::ExtendedLimitInfo::new();
            info.limit_kill_on_job_close();
            match win32job::Job::create_with_limit_info(&mut info) {
                Ok(job) => match job.assign_process(child.as_raw_handle() as isize) {
                    Ok(()) => {
                        log::info!("Python backend tied to job object (dies with this process, even on force-kill)");
                        self.job = Some(job);
                    }
                    Err(e) => log::warn!(
                        "Failed to assign Python backend to job object (force-kill orphan protection unavailable): {}",
                        e
                    ),
                },
                Err(e) => log::warn!(
                    "Failed to create job object (force-kill orphan protection unavailable): {}",
                    e
                ),
            }
        }

        // Spawn threads to read stdout and stderr
        if let Some(stdout) = child.stdout.take() {
            thread::spawn(move || {
                let reader = BufReader::new(stdout);
                for line in reader.lines() {
                    if let Ok(line) = line {
                        log::info!("[Python stdout] {}", line);
                    }
                }
            });
        }
        
        if let Some(stderr) = child.stderr.take() {
            thread::spawn(move || {
                let reader = BufReader::new(stderr);
                for line in reader.lines() {
                    if let Ok(line) = line {
                        log::error!("[Python stderr] {}", line);
                    }
                }
            });
        }
        
        self.process = Some(child);
        Ok(())
    }

    pub fn stop(&mut self) {
        if let Some(mut process) = self.process.take() {
            log::info!("Stopping Python backend process...");
            match process.kill() {
                Ok(_) => log::info!("Python backend stopped successfully"),
                Err(e) => log::error!("Failed to stop Python backend: {}", e),
            }
        }
        // Drop the job object alongside the process it was tied to -- by
        // this point the process is already killed above, so this is a
        // no-op in practice, just keeping the two fields' lifetimes in sync
        // rather than leaving a stale handle around until the next start().
        #[cfg(windows)]
        {
            self.job = None;
        }
    }

    pub fn is_running(&self) -> bool {
        if let Some(_process) = &self.process {
            // On Windows, try_wait to check if process has exited
            true // We don't have mutable access here, assume running
        } else {
            false
        }
    }
    
    pub fn check_status(&mut self) -> bool {
        if let Some(process) = &mut self.process {
            match process.try_wait() {
                Ok(Some(status)) => {
                    log::error!("Backend process exited with status: {:?}", status);
                    self.process = None;
                    false
                }
                Ok(None) => true, // Still running
                Err(e) => {
                    log::error!("Error checking backend status: {}", e);
                    false
                }
            }
        } else {
            false
        }
    }

    pub fn restart(&mut self, runtime: &PythonRuntime) -> Result<(), String> {
        log::info!("Restarting Python backend...");
        self.stop();
        std::thread::sleep(std::time::Duration::from_secs(2)); // Wait for graceful shutdown
        self.start(runtime)
    }
}

impl Drop for BackendProcess {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Where the Python interpreter and `backend`/`src` source live, and what
/// directory to launch `python -m backend.app` from.
#[derive(Debug, Clone)]
pub struct PythonRuntime {
    pub python_cmd: PathBuf,
    pub working_dir: PathBuf,
    /// Stable, per-user directory (Tauri's `app_data_dir`) for data that must
    /// survive reinstalls/updates -- chat sessions, the vector store, etc.
    /// Unlike `working_dir` (the NSIS resource dir in production), this is
    /// never wiped when the app is reinstalled or auto-updated.
    pub data_dir: PathBuf,
    /// True when running from a packaged/installed build (bundled Python
    /// runtime + resources found), false for a dev checkout falling back to
    /// `.venv`/system Python. Used to skip Uvicorn's dev-only `reload=True`
    /// supervisor in production -- nobody edits `backend`/`src` on an end
    /// user's machine, so it's pure startup overhead there.
    pub is_packaged: bool,
}

// Global backend state
pub struct BackendState {
    pub backend: Arc<Mutex<BackendProcess>>,
    pub runtime: PythonRuntime,
}

/// Check if the backend is responding by testing TCP connection to the port
fn wait_for_backend_ready(
    backend_mutex: Arc<Mutex<BackendProcess>>,
    port: u16,
    max_attempts: u32,
    delay_ms: u64
) -> bool {
    use std::net::TcpStream;
    use std::time::Duration;
    
    let addr = format!("127.0.0.1:{}", port);
    
    for attempt in 1..=max_attempts {
        // Check if process is still alive
        {
            let mut backend = backend_mutex.lock().unwrap();
            if !backend.check_status() {
                log::error!("Backend process has exited! Check Python errors above.");
                return false;
            }
        }
        
        log::info!("Checking backend readiness ({}/{})...", attempt, max_attempts);
        
        // Try to connect to the backend port
        match TcpStream::connect_timeout(
            &addr.parse().unwrap(),
            Duration::from_millis(500)
        ) {
            Ok(_) => {
                log::info!("Backend is accepting connections on port {}!", port);
                return true;
            }
            Err(e) => {
                log::debug!("Connection attempt {} failed: {}", attempt, e);
            }
        }
        
        if attempt < max_attempts {
            std::thread::sleep(Duration::from_millis(delay_ms));
        }
    }
    
    log::warn!("Backend did not start accepting connections after {} attempts", max_attempts);
    false
}

/// Locate the Python interpreter to run and the directory to run it from.
///
/// Production installs bundle a portable Python runtime alongside the
/// `backend`/`src` source under Tauri's resource directory (see
/// `bundle.resources` in tauri.conf.json) -- checked first, since a real
/// install has no repo checkout or dev `.venv` to fall back to. Dev mode
/// has no bundled resources, so it falls back to walking up from CWD to
/// find the repo checkout (identified by `backend/app.py`) and preferring
/// its `.venv` if present.
fn resolve_python_runtime(app: &AppHandle) -> Result<PythonRuntime, String> {
    // Resolved the same way in dev and production, so both modes persist
    // user data (sessions, vector store) to the same stable OS-appropriate
    // location instead of a path that happens to differ by run mode.
    let data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("Failed to resolve app data dir: {}", e))?;
    std::fs::create_dir_all(&data_dir)
        .map_err(|e| format!("Failed to create app data dir {:?}: {}", data_dir, e))?;
    log::info!("App data dir: {:?}", data_dir);

    match app.path().resource_dir() {
        Ok(resource_dir) => {
            let bundled_python = resource_dir.join("python-runtime").join("python.exe");
            let bundled_backend = resource_dir.join("backend").join("app.py");
            log::info!(
                "Resource dir: {:?} (python exists: {}, backend exists: {})",
                resource_dir, bundled_python.exists(), bundled_backend.exists()
            );
            if bundled_python.exists() && bundled_backend.exists() {
                log::info!("Using bundled Python runtime: {:?}", bundled_python);
                return Ok(PythonRuntime {
                    python_cmd: bundled_python,
                    working_dir: resource_dir,
                    data_dir,
                    is_packaged: true,
                });
            }
        }
        Err(e) => {
            log::info!("No resource dir available ({}), falling back to dev mode", e);
        }
    }

    let cwd = std::env::current_dir()
        .map_err(|e| format!("Failed to get current directory: {}", e))?;
    log::info!("Current working directory: {:?}", cwd);

    // `cargo run` sets CWD to the crate root (frontend/src-tauri), so the
    // project root is two levels up. Walk ancestors instead of hardcoding a
    // fixed depth, since that's what actually broke here previously.
    let project_root = cwd
        .ancestors()
        .find(|ancestor| ancestor.join("backend").join("app.py").exists())
        .map(|p| p.to_path_buf())
        .ok_or_else(|| format!("Could not find project root (backend/app.py) from CWD: {:?}", cwd))?;
    log::info!("Found project root: {:?}", project_root);

    let venv_python = if cfg!(windows) {
        project_root.join(".venv").join("Scripts").join("python.exe")
    } else {
        project_root.join(".venv").join("bin").join("python")
    };

    let python_cmd = if venv_python.exists() {
        log::info!("Using virtual environment Python: {:?}", venv_python);
        venv_python
    } else {
        log::info!("Virtual environment not found, using system Python");
        PathBuf::from("python")
    };

    Ok(PythonRuntime {
        python_cmd,
        working_dir: project_root,
        data_dir,
        is_packaged: false,
    })
}

pub fn init_backend(app: &AppHandle) -> Result<(), Box<dyn std::error::Error>> {
    let backend = BackendProcess::new(8000);
    let backend_mutex = Arc::new(Mutex::new(backend));

    // Resolve the Python runtime (bundled in production, dev .venv otherwise)
    let runtime = match resolve_python_runtime(app) {
        Ok(runtime) => runtime,
        Err(e) => {
            log::warn!("Could not resolve Python runtime: {}. Backend must be started manually.", e);
            log::warn!("Run 'python -m backend.app' from the project root directory.");
            // Still register state but with a placeholder runtime
            app.manage(BackendState {
                backend: backend_mutex,
                runtime: PythonRuntime {
                    python_cmd: PathBuf::from("python"),
                    working_dir: PathBuf::new(),
                    data_dir: PathBuf::new(),
                    is_packaged: false,
                },
            });
            return Ok(()); // Don't fail startup, just skip backend auto-start
        }
    };

    // Store backend state
    app.manage(BackendState {
        backend: backend_mutex.clone(),
        runtime: runtime.clone(),
    });

    // Start the Python backend and wait for it to become ready on a
    // background thread, not here. `init_backend` runs inside Tauri's
    // `.setup()`, which blocks the window's event loop until it returns --
    // waiting synchronously (up to 120s below) meant the window couldn't
    // process messages yet, which is exactly what makes Windows report
    // "Orion is not responding" regardless of how fast the backend actually
    // starts. The frontend already polls /health and shows its own
    // "backend not connected" banner, so there's nothing for setup() to wait
    // on here.
    thread::spawn(move || {
        log::info!("Auto-starting Python backend from: {:?}", runtime.working_dir);
        {
            let mut backend_guard = backend_mutex.lock().unwrap();
            if let Err(e) = backend_guard.start(&runtime) {
                log::error!("Failed to auto-start backend: {}. Please start manually with 'python -m backend.app'", e);
                return;
            }
        } // Release lock

        // Wait for backend to become ready (up to 120 seconds, checking every 500ms).
        // The old 15s budget gave up before the process could ever bind the
        // port, so the UI always reported "backend down" on first launch --
        // NOT because of model loading (SentenceTransformer/CrossEncoder are
        // already lazy, only loaded on the first real RAG query), but because
        // backend/app.py used to import the whole ML stack (torch,
        // sentence-transformers, chromadb, langchain) unconditionally at
        // module load, before uvicorn.run() was ever reached. That's now
        // fixed (those imports moved to lazy, function-local scope -- see
        // src/retrieval/embeddings.py, reranker.py, vector_store.py, and
        // src/core/ingest.py), so a generous budget here is now mostly
        // insurance against slow disk/AV-scan cold starts on a fresh
        // install, not the expected common case.
        log::info!("Waiting for backend to become ready...");
        if wait_for_backend_ready(backend_mutex.clone(), 8000, 240, 500) {
            log::info!("Backend is ready!");
        } else {
            log::warn!("Backend may not be fully ready. Check the console for Python errors.");
        }
    });

    Ok(())
}

pub fn cleanup_backend(app: &AppHandle) {
    if let Some(state) = app.try_state::<BackendState>() {
        let mut backend = state.backend.lock().unwrap();
        backend.stop();
        log::info!("Backend cleanup completed");
    }
}
