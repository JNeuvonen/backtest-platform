import { ENV } from "../utils/constants";
import { retrieveEnvVar } from "../utils/tauri";

const API = {
  tables: "/tables",
};

export async function tablesUrl() {
  const baseUrl = await retrieveEnvVar(ENV.base_url);
  console.log(baseUrl);
  const url = baseUrl + API.tables;
  return url;
}
