# Tauri Integration Roadmap — Orion Desktop App

**Purpose**: Wrap Orion in a native desktop application with system tray, window management, and native OS integration.  
**Date**: February 2026  
**Status**: ✅ Complete (2026-07-29) — sidecar backend spawn, system tray, IPC commands, native dialogs/notifications all shipped in v1. Auto-update and production packaging of the backend sidecar remain open (see README Roadmap).
---

## What is Tauri?

Tauri is a toolkit for building lightweight, secure desktop applications using web technologies (HTML/CSS/JS) for the frontend and Rust for the backend/native layer. Think of it as a lighter alternative to Electron.

**Why Tauri for Orion?**
| Feature | Benefit for Orion |
|---------|-------------------|
| **Small binary size** | ~5-10MB vs 150MB+ for Electron |
| **Native system tray** | Built-in support, no hacks needed |
| **Low memory footprint** | Important since we're also running LLMs |
| **Security-first** | Sandboxed by default, explicit permissions |
| **Cross-platform** | Windows, macOS, Linux from same codebase |
| **Rust backend** | Can call Python subprocess or HTTP endpoints |

---

## Answers to Your Questions

### 1. Do I need to pip install or install any dependencies?

**No pip installs for Tauri itself.** Tauri is a Rust/Node ecosystem tool.

**Required installations:**

| Tool | Purpose | Installation |
|------|---------|--------------|
| **Rust** | Tauri's core runtime | `winget install Rust.Rustup` or [rustup.rs](https://rustup.rs) |
| **Node.js** | Tauri CLI and build tools | Already have (for Svelte frontend) |
| **Tauri CLI** | Project scaffolding & builds | `npm install -D @tauri-apps/cli` |
| **WebView2** | Windows rendering engine | Usually pre-installed on Win10/11 |

**One-time setup (Windows):**
```powershell
# Install Rust (if not installed)
winget install Rust.Rustup

# Verify Rust installation
rustc --version

# In frontend directory, add Tauri CLI
cd frontend
npm install -D @tauri-apps/cli @tauri-apps/api
```

### 2. Does Tauri handle the installer?

**Yes!** Tauri has built-in bundler that creates platform-specific installers:

| Platform | Output |
|----------|--------|
| Windows | `.msi` installer + `.exe` (NSIS optional) |
| macOS | `.dmg` + `.app` bundle |
| Linux | `.deb`, `.AppImage`, `.rpm` |

```bash
# Build release with installer
npm run tauri build
# Output: src-tauri/target/release/bundle/
```

### 3. How do I handle updates? (Future)

Tauri has an **updater plugin** for auto-updates:

```rust
// In tauri.conf.json
"updater": {
  "active": true,
  "endpoints": ["https://your-server.com/updates/{{target}}/{{current_version}}"],
  "dialog": true,   // Show "Update available" dialog
  "pubkey": "YOUR_PUBLIC_KEY"
}
```

**Update flow:**
1. Host update manifests + binaries on GitHub Releases or your server
2. App checks endpoint on startup (configurable)
3. If new version available → prompt user → download → restart

**We'll skip this for now** — focus on core integration first.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        ORION DESKTOP                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐     ┌─────────────────────────────────┐    │
│  │   System Tray   │     │       Main Window (WebView)     │    │
│  │   (Rust/Tauri)  │     │         Svelte Frontend         │    │
│  │                 │     │                                 │    │
│  │  • Quick Query  │     │  • Chat UI                      │    │
│  │  • Start/Stop   │     │  • Settings                     │    │
│  │  • Quit         │     │  • Ingestion Status             │    │
│  └────────┬────────┘     └────────────────┬────────────────┘    │
│           │                               │                     │
│           │         Tauri Commands        │                     │
│           │      (IPC: invoke/listen)     │                     │
│           └───────────────┬───────────────┘                     │
│                           │                                     │
│  ┌────────────────────────▼─────────────────────────────────┐   │
│  │                    Rust Backend                          │   │
│  │                   (src-tauri/)                           │   │
│  │                                                          │   │
│  │  • Spawn/manage Python backend process                   │   │
│  │  • System tray management                                │   │
│  │  • Window management (show/hide/minimize to tray)        │   │
│  │  • Native file dialogs                                   │   │
│  │  • OS notifications                                      │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                     │
│                    HTTP/WebSocket                               │
│                           │                                     │
│  ┌────────────────────────▼─────────────────────────────────┐   │
│  │              Python Backend (FastAPI)                    │   │
│  │                  (backend/app.py)                        │   │
│  │                                                          │   │
│  │  • RAG Pipeline                                          │   │
│  │  • LLM Communication (Ollama)                            │   │
│  │  • Watchdog                                              │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Key insight**: Tauri doesn't replace Python — it wraps the frontend and manages the Python backend as a subprocess.

---

## Directory Structure

```
Orion/
├── frontend/                    # Existing Svelte app
│   ├── src/
│   ├── src-tauri/              # 🆕 TAURI CORE (created by tauri init)
│   │   ├── Cargo.toml          # Rust dependencies
│   │   ├── tauri.conf.json     # Tauri configuration
│   │   ├── build.rs            # Build script
│   │   ├── icons/              # App icons (all sizes)
│   │   │   ├── icon.ico        # Windows
│   │   │   ├── icon.icns       # macOS
│   │   │   ├── icon.png        # Linux
│   │   │   ├── 32x32.png
│   │   │   ├── 128x128.png
│   │   │   └── ...
│   │   └── src/
│   │       ├── main.rs         # Entry point
│   │       ├── lib.rs          # Command definitions
│   │       ├── tray.rs         # 🆕 System tray logic
│   │       ├── backend.rs      # 🆕 Python process management
│   │       └── commands.rs     # 🆕 IPC commands (Svelte ↔ Rust)
│   ├── package.json            # Add @tauri-apps/cli, @tauri-apps/api
│   └── vite.config.ts          # May need minor tweaks
│
├── backend/                     # Existing FastAPI (unchanged)
├── src/                         # Existing Python RAG code (unchanged)
└── ...
```

**Why inside `frontend/`?**  
Tauri expects to wrap a web app. By placing `src-tauri/` inside `frontend/`, Tauri can directly serve the Svelte build output. This is the standard Tauri + Vite/Svelte pattern.

---

## Core Files & Their Roles

### 1. `tauri.conf.json` — Central Configuration

```json
{
  "$schema": "https://schema.tauri.app/config/2",
  "productName": "Orion",
  "version": "1.0.0",
  "identifier": "com.orion.app",
  "build": {
    "beforeBuildCommand": "npm run build",
    "beforeDevCommand": "npm run dev",
    "devUrl": "http://localhost:5173",
    "frontendDist": "../build"
  },
  "app": {
    "withGlobalTauri": true,
    "windows": [
      {
        "title": "Orion",
        "width": 1200,
        "height": 800,
        "resizable": true,
        "fullscreen": false,
        "visible": true,
        "decorations": true,
        "transparent": false
      }
    ],
    "security": {
      "csp": null
    },
    "trayIcon": {
      "iconPath": "icons/icon.png",
      "iconAsTemplate": true
    }
  },
  "bundle": {
    "active": true,
    "targets": "all",
    "icon": [
      "icons/32x32.png",
      "icons/128x128.png",
      "icons/icon.ico",
      "icons/icon.icns"
    ],
    "windows": {
      "wix": {
        "language": "en-US"
      }
    }
  }
}
```

### 2. `main.rs` — Application Entry Point

```rust
// src-tauri/src/main.rs
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod tray;
mod backend;
mod commands;

use tauri::Manager;

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            // Initialize system tray
            tray::create_tray(app)?;
            
            // Start Python backend
            backend::spawn_backend(app)?;
            
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_backend_status,
            commands::restart_backend,
            commands::open_knowledge_folder,
            commands::show_notification,
        ])
        .on_window_event(|window, event| {
            // Minimize to tray instead of closing
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                window.hide().unwrap();
                api.prevent_close();
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

### 3. `tray.rs` — System Tray Implementation

```rust
// src-tauri/src/tray.rs
use tauri::{
    AppHandle, CustomMenuItem, Manager, SystemTray, SystemTrayEvent, 
    SystemTrayMenu, SystemTrayMenuItem,
};

pub fn create_tray(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let quit = CustomMenuItem::new("quit".to_string(), "Quit Orion");
    let show = CustomMenuItem::new("show".to_string(), "Show Window");
    let hide = CustomMenuItem::new("hide".to_string(), "Hide Window");
    let separator = SystemTrayMenuItem::Separator;
    
    let tray_menu = SystemTrayMenu::new()
        .add_item(show)
        .add_item(hide)
        .add_native_item(separator)
        .add_item(quit);
    
    let tray = SystemTray::new().with_menu(tray_menu);
    
    // Handle tray events
    app.tray_handle().set_menu(tray_menu)?;
    
    Ok(())
}

pub fn handle_tray_event(app: &AppHandle, event: SystemTrayEvent) {
    match event {
        SystemTrayEvent::LeftClick { .. } => {
            // Toggle window visibility on left click
            if let Some(window) = app.get_window("main") {
                if window.is_visible().unwrap_or(false) {
                    window.hide().unwrap();
                } else {
                    window.show().unwrap();
                    window.set_focus().unwrap();
                }
            }
        }
        SystemTrayEvent::MenuItemClick { id, .. } => match id.as_str() {
            "quit" => {
                // Cleanup and exit
                std::process::exit(0);
            }
            "show" => {
                if let Some(window) = app.get_window("main") {
                    window.show().unwrap();
                    window.set_focus().unwrap();
                }
            }
            "hide" => {
                if let Some(window) = app.get_window("main") {
                    window.hide().unwrap();
                }
            }
            _ => {}
        },
        _ => {}
    }
}
```

### 4. `backend.rs` — Python Process Manager

```rust
// src-tauri/src/backend.rs
use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::AppHandle;

pub struct BackendProcess {
    process: Option<Child>,
}

impl BackendProcess {
    pub fn new() -> Self {
        Self { process: None }
    }
    
    pub fn start(&mut self, app_dir: &str) -> Result<(), String> {
        // Path to Python backend
        let backend_script = format!("{}/run.py", app_dir);
        
        // Spawn Python backend
        let child = Command::new("python")
            .arg(&backend_script)
            .arg("--host").arg("127.0.0.1")
            .arg("--port").arg("8000")
            .spawn()
            .map_err(|e| format!("Failed to start backend: {}", e))?;
        
        self.process = Some(child);
        Ok(())
    }
    
    pub fn stop(&mut self) {
        if let Some(mut process) = self.process.take() {
            let _ = process.kill();
        }
    }
    
    pub fn is_running(&self) -> bool {
        self.process.is_some()
    }
}

// Global state for backend process
pub static BACKEND: Mutex<Option<BackendProcess>> = Mutex::new(None);

pub fn spawn_backend(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let app_dir = app.path_resolver()
        .app_data_dir()
        .ok_or("Failed to get app directory")?;
    
    let mut backend = BackendProcess::new();
    backend.start(app_dir.to_str().unwrap())?;
    
    *BACKEND.lock().unwrap() = Some(backend);
    
    Ok(())
}
```

### 5. `commands.rs` — IPC Bridge (Svelte ↔ Rust)

```rust
// src-tauri/src/commands.rs
use tauri::command;

#[command]
pub async fn get_backend_status() -> Result<String, String> {
    // Check if Python backend is responding
    let client = reqwest::Client::new();
    match client.get("http://127.0.0.1:8000/api/health").send().await {
        Ok(response) if response.status().is_success() => Ok("running".to_string()),
        _ => Ok("stopped".to_string()),
    }
}

#[command]
pub async fn restart_backend() -> Result<(), String> {
    use crate::backend::BACKEND;
    
    let mut backend_guard = BACKEND.lock().map_err(|e| e.to_string())?;
    if let Some(backend) = backend_guard.as_mut() {
        backend.stop();
        backend.start(".")?;
    }
    Ok(())
}

#[command]
pub async fn open_knowledge_folder(path: String) -> Result<(), String> {
    open::that(&path).map_err(|e| e.to_string())
}

#[command]
pub fn show_notification(title: String, body: String) -> Result<(), String> {
    // Use native notification
    notify_rust::Notification::new()
        .summary(&title)
        .body(&body)
        .show()
        .map_err(|e| e.to_string())?;
    Ok(())
}
```

### 6. Frontend Integration (`lib/tauri.ts`)

```typescript
// frontend/src/lib/tauri.ts
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';

// Check if running in Tauri
export const isTauri = () => '__TAURI__' in window;

// Call Rust commands
export async function getBackendStatus(): Promise<string> {
  if (!isTauri()) return 'browser-mode';
  return await invoke('get_backend_status');
}

export async function restartBackend(): Promise<void> {
  if (!isTauri()) return;
  await invoke('restart_backend');
}

export async function openFolder(path: string): Promise<void> {
  if (!isTauri()) return;
  await invoke('open_knowledge_folder', { path });
}

export async function showNotification(title: string, body: string): Promise<void> {
  if (!isTauri()) {
    // Fallback to browser notification
    new Notification(title, { body });
    return;
  }
  await invoke('show_notification', { title, body });
}

// Listen for events from Rust
export function onBackendEvent(callback: (event: any) => void) {
  if (!isTauri()) return () => {};
  return listen('backend-event', callback);
}
```

---

## Implementation Phases

### Phase 1: Scaffolding (Day 1)
```
□ Install Rust via rustup
□ Run `npm install -D @tauri-apps/cli @tauri-apps/api` in frontend/
□ Run `npx tauri init` to scaffold src-tauri/
□ Configure tauri.conf.json (window settings, app name)
□ Test basic build: `npx tauri dev`
```

### Phase 2: System Tray (Day 1-2)
```
□ Create tray.rs with menu items
□ Implement show/hide/quit actions
□ Handle minimize-to-tray on window close
□ Add tray icon (create icon assets)
□ Left-click toggles window
```

### Phase 3: Backend Management (Day 2-3)
```
□ Create backend.rs subprocess manager
□ Auto-start Python backend on app launch
□ Graceful shutdown on app quit
□ Health check polling
□ Restart command
```

### Phase 4: IPC Integration (Day 3-4)
```
□ Create commands.rs with Tauri commands
□ Create frontend/src/lib/tauri.ts wrapper
□ Wire up status indicators in UI
□ Native file dialogs for folder selection
□ Native notifications for ingestion complete
```

### Phase 5: Polish & Testing (Day 4-5)
```
□ Window state persistence (size, position)
□ Auto-start on system boot (optional)
□ Error handling and logging
□ Test on Windows (primary target)
□ Build installer: `npx tauri build`
```

---

## Cargo.toml Dependencies

```toml
# src-tauri/Cargo.toml
[package]
name = "orion"
version = "1.0.0"
edition = "2021"

[build-dependencies]
tauri-build = { version = "2", features = [] }

[dependencies]
tauri = { version = "2", features = ["tray-icon", "devtools"] }
tauri-plugin-shell = "2"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
reqwest = { version = "0.11", features = ["json"] }
tokio = { version = "1", features = ["full"] }
notify-rust = "4"  # Native notifications
open = "5"         # Open files/folders in default app

[features]
default = ["custom-protocol"]
custom-protocol = ["tauri/custom-protocol"]
```

---

## Key Differences from Electron

| Aspect | Electron | Tauri |
|--------|----------|-------|
| Runtime | Bundles Chromium + Node.js | Uses system WebView + Rust |
| Binary size | 150-200MB | 5-15MB |
| Memory | 200MB+ base | 30-50MB base |
| Backend | JavaScript (Node.js) | Rust (can spawn Python) |
| Security | Manual sandboxing | Sandboxed by default |
| IPC | `ipcMain`/`ipcRenderer` | `invoke`/`listen` |

---

## Common Gotchas & Tips

### 1. WebView2 on Windows
Windows 10/11 ships with Edge WebView2. For older Windows, Tauri can bundle the WebView2 bootstrapper in the installer.

### 2. Python Path in Production
In dev, Python is in PATH. In production:
- Option A: Require Python pre-installed, document in README
- Option B: Bundle Python with PyInstaller first, then wrap with Tauri
- **Recommended for Orion**: Option A (users running local LLMs likely have Python)

### 3. Development Workflow
```bash
# Terminal 1: Run Svelte dev server
cd frontend && npm run dev

# Terminal 2: Run Tauri dev (wraps the Svelte server)
cd frontend && npx tauri dev

# For Python backend (separate, as usual)
# Terminal 3: python run.py
```

### 4. Hot Reload
Tauri dev mode hot-reloads frontend changes. Rust changes require restart.

---

## File Checklist for MVP

```
✅ = Must have for MVP
🔜 = Nice to have
⏭️ = Future

src-tauri/
├── Cargo.toml              ✅ Dependencies
├── tauri.conf.json         ✅ App config
├── build.rs                ✅ Build script (auto-generated)
├── icons/                  ✅ App icons
│   ├── icon.ico            ✅ Windows
│   ├── icon.png            ✅ Tray icon
│   └── ...
└── src/
    ├── main.rs             ✅ Entry point
    ├── lib.rs              ✅ Module exports
    ├── tray.rs             ✅ System tray
    ├── backend.rs          ✅ Python process management
    ├── commands.rs         ✅ IPC commands
    ├── window.rs           🔜 Window state persistence
    └── updater.rs          ⏭️ Auto-update logic
```

---

## Next Steps

1. **Confirm prerequisites**: Do you have Rust installed? Run `rustc --version`
2. **Start Phase 1**: I'll scaffold Tauri in the frontend directory
3. **Iterate**: Build incrementally, test each phase

Ready to proceed with Phase 1 scaffolding?
