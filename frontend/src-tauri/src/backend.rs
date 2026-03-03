// src-tauri/src/backend.rs
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::path::PathBuf;
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

    pub fn start(&mut self, app_dir: PathBuf) -> Result<(), String> {
        if self.is_running() {
            return Ok(()); // Already running
        }

        // Navigate to project root (parent of frontend)
        let project_root = app_dir
            .parent()
            .ok_or("Failed to get project root")?;

        log::info!("Starting Python backend from: {:?}", project_root);
        log::info!("Backend will run on port: {}", self.port);

        // Start Python backend with run.py
        let child = Command::new("python")
            .arg("run.py")
            .current_dir(project_root)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to start Python backend: {}. Make sure Python is installed and run.py exists.", e))?;

        log::info!("Python backend process started with PID: {:?}", child.id());
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
        if let Some(process) = &self.process {
            // Check if process is still alive
            match process.id() {
                0 => false, // Process has exited
                _ => true,
            }
        } else {
            false
        }
    }

    pub fn restart(&mut self, app_dir: PathBuf) -> Result<(), String> {
        log::info!("Restarting Python backend...");
        self.stop();
        std::thread::sleep(std::time::Duration::from_secs(2)); // Wait for graceful shutdown
        self.start(app_dir)
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
}

pub fn init_backend(app: &AppHandle) -> Result<(), Box<dyn std::error::Error>> {
    let backend = BackendProcess::new(8000);
    let backend_mutex = Arc::new(Mutex::new(backend));

    // Store backend in app state
    app.manage(BackendState {
        backend: backend_mutex.clone(),
    });

    // Get app directory (where src-tauri is located)
    let app_dir = app.path()
        .app_config_dir()
        .map_err(|e| format!("Failed to get app directory: {}", e))?;

    log::info!("App directory: {:?}", app_dir);

    // Start the backend
    let mut backend_guard = backend_mutex.lock().unwrap();
    backend_guard.start(app_dir)?;

    Ok(())
}

pub fn cleanup_backend(app: &AppHandle) {
    if let Some(state) = app.try_state::<BackendState>() {
        let mut backend = state.backend.lock().unwrap();
        backend.stop();
        log::info!("Backend cleanup completed");
    }
}
