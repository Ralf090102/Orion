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

    pub fn start(&mut self, project_root: PathBuf) -> Result<(), String> {
        if self.is_running() {
            return Ok(()); // Already running
        }

        log::info!("Starting Python backend from: {:?}", project_root);
        log::info!("Backend will run on port: {}", self.port);

        // Determine the Python executable path
        // Prefer the virtual environment if it exists
        let venv_python = if cfg!(windows) {
            project_root.join(".venv").join("Scripts").join("python.exe")
        } else {
            project_root.join(".venv").join("bin").join("python")
        };
        
        let python_cmd = if venv_python.exists() {
            log::info!("Using virtual environment Python: {:?}", venv_python);
            venv_python.to_string_lossy().to_string()
        } else {
            log::info!("Virtual environment not found, using system Python");
            "python".to_string()
        };

        // Start Python backend with backend/app.py (FastAPI server)
        // Pipe output so we can log it
        let mut child = Command::new(&python_cmd)
            .args(["-m", "backend.app"])  // Run as module: python -m backend.app
            .current_dir(&project_root)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to start Python backend: {}. Make sure Python is installed and backend/app.py exists at {:?}", e, project_root))?;

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

    pub fn restart(&mut self, project_root: PathBuf) -> Result<(), String> {
        log::info!("Restarting Python backend...");
        self.stop();
        std::thread::sleep(std::time::Duration::from_secs(2)); // Wait for graceful shutdown
        self.start(project_root)
    }
}

impl Drop for BackendProcess {
    fn drop(&mut self) {
        self.stop();
    }
}

// Global backend state
pub struct BackendState {
    pub backend: Arc<Mutex<BackendProcess>>,
    pub project_root: PathBuf,
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

/// Get the project root directory
/// In dev mode: uses current working directory's parent (frontend/../ = project root)
/// In production: would use bundled resources or require Python pre-started
fn get_project_root() -> Result<PathBuf, String> {
    // `cargo run` sets CWD to the crate root (frontend/src-tauri), so the
    // project root is two levels up. Walk ancestors instead of hardcoding a
    // fixed depth, since that's what actually broke here.
    let cwd = std::env::current_dir()
        .map_err(|e| format!("Failed to get current directory: {}", e))?;

    log::info!("Current working directory: {:?}", cwd);

    for ancestor in cwd.ancestors() {
        if ancestor.join("backend").join("app.py").exists() {
            log::info!("Found project root: {:?}", ancestor);
            return Ok(ancestor.to_path_buf());
        }
    }

    Err(format!("Could not find project root (backend/app.py) from CWD: {:?}", cwd))
}

pub fn init_backend(app: &AppHandle) -> Result<(), Box<dyn std::error::Error>> {
    let backend = BackendProcess::new(8000);
    let backend_mutex = Arc::new(Mutex::new(backend));

    // Find project root
    let project_root = match get_project_root() {
        Ok(path) => path,
        Err(e) => {
            log::warn!("Could not find project root: {}. Backend must be started manually.", e);
            log::warn!("Run 'python -m backend.app' from the project root directory.");
            // Still register state but with a placeholder path
            app.manage(BackendState {
                backend: backend_mutex,
                project_root: PathBuf::new(),
            });
            return Ok(()); // Don't fail startup, just skip backend auto-start
        }
    };

    // Store backend state
    app.manage(BackendState {
        backend: backend_mutex.clone(),
        project_root: project_root.clone(),
    });

    // Auto-start the Python backend
    log::info!("Auto-starting Python backend from: {:?}", project_root);
    {
        let mut backend_guard = backend_mutex.lock().unwrap();
        if let Err(e) = backend_guard.start(project_root) {
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
