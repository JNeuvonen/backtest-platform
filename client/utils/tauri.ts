import { invoke } from "@tauri-apps/api";
import { TAURI_COMMANDS } from "./constants";

export async function retrieveEnvVar(envVar: string) {
  try {
    const url = await invoke(TAURI_COMMANDS.fetch_env, { key: envVar });
    return url;
  } catch (error) {
    console.error("Error getting backend URL:", error);
    return "";
  }
}
