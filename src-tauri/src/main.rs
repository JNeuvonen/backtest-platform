use std::{
    fs,
    path::Path,
    process::{Command, Stdio},
};

#[tauri::command]
fn fetch_env(key: String) -> Option<String> {
    std::env::var(key).ok()
}

fn main() {
    dotenv::dotenv().ok();
    tauri::Builder::default()
        .setup(|app| {
            let app_data_path = app
                .path_resolver()
                .app_data_dir()
                .expect("Failed to find app data directory")
                .to_str()
                .expect("Failed to convert app data path to string")
                .to_string();

            if !Path::new(&app_data_path).exists() {
                //init app data dir in prod
                fs::create_dir_all(&app_data_path).expect("Failed to create app data directory");
            }
            if cfg!(debug_assertions) {
                let _fastapi_server = Command::new("python")
                    .current_dir("../pyserver/src")
                    .arg("-m")
                    .arg("uvicorn")
                    .arg("server:app") // Replace 'server:app' with your actual module and app name
                    .arg("--reload")
                    .env("APP_DATA_PATH", &app_data_path)
                    .spawn()
                    .expect("Failed to start FastAPI server");
            } else {
                let binary_path = app
                    .path_resolver()
                    .resource_dir()
                    .unwrap()
                    .join("binaries/build/aarch64-apple-darwin/debug/install/pyserver")
                    .to_str()
                    .unwrap()
                    .to_string();

                let _fastapi_server = Command::new(binary_path)
                    .stdout(Stdio::piped())
                    .env("APP_DATA_PATH", &app_data_path)
                    .spawn()
                    .expect("Failed to start FastAPI server");
            }

            // Start the FastAPI server using the correct path

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![fetch_env])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
