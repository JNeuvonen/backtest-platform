import { invoke } from "@tauri-apps/api/tauri";
import { TAURI_COMMANDS } from "./constants";

export async function fetchEnvVar(envVar: string) {
  try {
    const url = await invoke(TAURI_COMMANDS.fetch_env, { key: envVar });
    return url;
  } catch (error) {
    console.error("Error getting backend URL:", error);
    return "";
  }
}

export async function fetchPlatform() {
  try {
    const url = await invoke(TAURI_COMMANDS.fetch_platform);
    return url;
  } catch (error) {
    console.error("Error getting backend URL:", error);
    return "";
  }
}
