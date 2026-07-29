mod tray;
mod backend;
mod commands;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  use tauri::Manager;
  
  tauri::Builder::default()
    .plugin(tauri_plugin_dialog::init())
    .plugin(tauri_plugin_notification::init())
    .setup(|app| {
      if cfg!(debug_assertions) {
        app.handle().plugin(
          tauri_plugin_log::Builder::default()
            .level(log::LevelFilter::Info)
            .build(),
        )?;
      }

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
