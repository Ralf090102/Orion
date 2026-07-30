// src-tauri/src/commands.rs
use crate::backend::BackendState;
use tauri::State;

#[tauri::command]
pub async fn get_backend_status() -> Result<String, String> {
    log::info!("Checking backend status...");
    
    // Try to ping the health endpoint
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .map_err(|e| e.to_string())?;

    match client.get("http://127.0.0.1:8000/health").send().await {
        Ok(response) if response.status().is_success() => {
            log::info!("Backend is running");
            Ok("running".to_string())
        }
        Ok(response) => {
            log::warn!("Backend returned unexpected status: {}", response.status());
            Ok("error".to_string())
        }
        Err(e) => {
            log::warn!("Backend not responding: {}", e);
            Ok("stopped".to_string())
        }
    }
}

#[tauri::command]
pub async fn restart_backend(
    state: State<'_, BackendState>,
) -> Result<(), String> {
    log::info!("Restart backend command received");

    let runtime = state.runtime.clone();
    if runtime.working_dir.as_os_str().is_empty() {
        return Err("Python runtime not configured. Please start backend manually.".to_string());
    }

    let mut backend = state.backend.lock().unwrap();
    backend.restart(&runtime)?;

    log::info!("Backend restart initiated");
    Ok(())
}

#[tauri::command]
pub async fn stop_backend(state: State<'_, BackendState>) -> Result<(), String> {
    log::info!("Stop backend command received");
    
    let mut backend = state.backend.lock().unwrap();
    backend.stop();
    
    log::info!("Backend stopped");
    Ok(())
}

#[tauri::command]
pub async fn start_backend(
    state: State<'_, BackendState>,
) -> Result<(), String> {
    log::info!("Start backend command received");

    let runtime = state.runtime.clone();
    if runtime.working_dir.as_os_str().is_empty() {
        return Err("Python runtime not configured. Please start backend manually.".to_string());
    }

    let mut backend = state.backend.lock().unwrap();
    backend.start(&runtime)?;

    log::info!("Backend start initiated");
    Ok(())
}

#[tauri::command]
pub fn open_folder(path: String) -> Result<(), String> {
    log::info!("Opening folder: {}", path);
    
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("explorer")
            .arg(&path)
            .spawn()
            .map_err(|e| e.to_string())?;
    }

    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(&path)
            .spawn()
            .map_err(|e| e.to_string())?;
    }

    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open")
            .arg(&path)
            .spawn()
            .map_err(|e| e.to_string())?;
    }

    Ok(())
}
