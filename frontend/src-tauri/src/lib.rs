mod tray;
mod backend;
mod commands;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  use tauri::Manager;
  
  tauri::Builder::default()
    .plugin(tauri_plugin_dialog::init())
    .plugin(tauri_plugin_notification::init())
    .plugin(tauri_plugin_updater::Builder::new().build())
    .plugin(tauri_plugin_process::init())
    .setup(|app| {
      use tauri_plugin_log::{Target, TargetKind, RotationStrategy};

      // Was gated behind `cfg!(debug_assertions)`, which meant release
      // builds installed no logger at all -- every log::info!/error! call
      // anywhere in this crate (including the Python child's relayed
      // stdout/stderr below) was a silent no-op in a signed, shipped build.
      // Diagnosing a live RAG-groundedness bug 2026-09-11 was measurably
      // slowed by having no persistent logs at all; see Eru's
      // Orion-Roadmap.md for the full decision record. Logs now land in a
      // `logs/` subfolder of the same stable app-data dir Python's own log
      // file and `open_folder` use, so there's one folder to open, not two
      // OS-convention log locations.
      let data_dir = backend::resolve_data_dir(app.handle())
        .unwrap_or_else(|e| {
          log::error!("Failed to resolve data dir for logging, falling back to temp dir: {}", e);
          std::env::temp_dir()
        });
      let log_dir = data_dir.join("logs");

      let level = if cfg!(debug_assertions) { log::LevelFilter::Debug } else { log::LevelFilter::Info };
      app.handle().plugin(
        tauri_plugin_log::Builder::new()
          // Builder's own defaults are [Stdout, LogDir{file_name: None}], and
          // .target() appends rather than replaces -- without clearing them
          // first, the plugin's default LogDir target would keep silently
          // writing a *second* log file to its own OS-convention location
          // (not this data_dir), which the "Open Logs Folder" button would
          // never reveal. Confirmed against the actual tauri-plugin-log
          // 2.8.0 source (Builder::default_targets()), not assumed.
          .clear_targets()
          .target(Target::new(TargetKind::Folder { path: log_dir, file_name: Some("orion-shell".into()) }))
          .target(Target::new(TargetKind::Stdout))
          .level(level)
          .max_file_size(10_000_000) // crate default is 40_000 bytes -- rotates almost immediately, too aggressive
          .rotation_strategy(RotationStrategy::KeepOne) // bounded: ~20MB total, not unbounded growth
          .build(),
      )?;

      // Initialize system tray
      tray::create_tray(app.handle())?;

      // Initialize and start Python backend
      backend::init_backend(app.handle())?;

      Ok(())
    })
    .invoke_handler(tauri::generate_handler![
      commands::get_backend_status,
      commands::restart_backend,
      commands::stop_backend,
      commands::start_backend,
      commands::open_folder,
      commands::get_logs_dir,
    ])
    .on_window_event(|window, event| {
      // Minimize to tray instead of closing
      if let tauri::WindowEvent::CloseRequested { api, .. } = event {
        window.hide().unwrap();
        api.prevent_close();
      }
    })
    .build(tauri::generate_context!())
    .expect("error while building tauri application")
    .run(|app_handle, event| {
      // Handle app exit - cleanup backend
      if let tauri::RunEvent::ExitRequested { .. } = event {
        log::info!("App exit requested, cleaning up...");
        backend::cleanup_backend(app_handle);
      }
    });
}
