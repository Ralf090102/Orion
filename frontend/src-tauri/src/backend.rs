// src-tauri/src/backend.rs
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::path::PathBuf;
use std::io::{BufRead, BufReader};
use std::thread;
use tauri::{AppHandle, Manager};

#[derive(Debug)]
pub struct BackendProcess {
    process: Option<Child>,
    port: u16,
}

impl BackendProcess {
    pub fn new(port: u16) -> Self {
        Self {
            process: None,
            port,
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
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!(
                "Failed to start Python backend: {}. Tried to run {:?} -m backend.app from {:?}",
                e, runtime.python_cmd, runtime.working_dir
            ))?;

        log::info!("Python backend process started with PID: {:?}", child.id());
        
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

    // Auto-start the Python backend
    log::info!("Auto-starting Python backend from: {:?}", runtime.working_dir);
    {
        let mut backend_guard = backend_mutex.lock().unwrap();
        if let Err(e) = backend_guard.start(&runtime) {
            log::error!("Failed to auto-start backend: {}. Please start manually with 'python -m backend.app'", e);
            return Ok(());
        }
    } // Release lock
    
    // Wait for backend to become ready (up to 120 seconds, checking every 500ms).
    // Cold start loads the embedding + reranker models before Uvicorn binds the
    // port, which measured ~65s on a cold process — the old 15s budget gave up
    // long before that, so the UI always reported "backend down" on first launch.
    log::info!("Waiting for backend to become ready...");
    if wait_for_backend_ready(backend_mutex.clone(), 8000, 240, 500) {
        log::info!("Backend is ready!");
    } else {
        log::warn!("Backend may not be fully ready. Check the console for Python errors.");
    }

    Ok(())
}

pub fn cleanup_backend(app: &AppHandle) {
    if let Some(state) = app.try_state::<BackendState>() {
        let mut backend = state.backend.lock().unwrap();
        backend.stop();
        log::info!("Backend cleanup completed");
    }
}
