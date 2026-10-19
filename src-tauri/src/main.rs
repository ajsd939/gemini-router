// Hide the console window on Windows release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod anthropic;
mod commands;
mod config;
mod gemini;
mod inject;
mod keystore;
mod logging;
mod openai;
mod responses;
mod router;
mod server;
mod state;

use std::sync::Arc;

use tauri::Manager;

use state::AppState;

/// Write a crash report to `%TEMP%\gemini-router-crash.log`.
///
/// Release builds use the Windows GUI subsystem, so a panic during startup has
/// no visible console. This makes the otherwise-silent "the window flashed and
/// closed" failure diagnosable.
fn install_panic_logger() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let path = std::env::temp_dir().join("gemini-router-crash.log");
        let report = format!(
            "Gemini Router crashed at {}\n{info}\n\nBacktrace:\n{}\n",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
            std::backtrace::Backtrace::force_capture()
        );
        let _ = std::fs::write(&path, &report);
        eprintln!("{report}");
        default_hook(info);
    }));
}

fn main() {
    install_panic_logger();
    tauri::Builder::default()
        // A second launch focuses the existing window instead of trying to
        // create a second WebView2 against the same user-data folder (which
        // fails with "The requested resource is in use" and closes the app).
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.show();
                let _ = window.unminimize();
                let _ = window.set_focus();
            }
        }))
        .setup(|app| {
            // State is created here so it can hold the AppHandle for event emission.
            let state = Arc::new(AppState::new(app.handle().clone())?);
            app.manage(state);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_snapshot,
            commands::refresh,
            commands::set_mode,
            commands::set_config_json,
            commands::reload_config,
            commands::add_key_slot,
            commands::save_key,
            commands::clear_key,
            commands::test_key,
            commands::send_prompt,
            commands::start_server,
            commands::stop_server,
            commands::update_server_settings,
            commands::add_access_token,
            commands::remove_access_token,
            commands::regenerate_access_token,
            commands::inject_targets,
            commands::inject_tool,
            commands::clear_log,
        ])
        .run(tauri::generate_context!())
        .expect("error while running the Gemini Router application");
}
