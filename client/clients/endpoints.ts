import { CONSTANTS } from "../utils/constants";

const API = {
  tables: "/tables",
};

const BASE_URL = CONSTANTS.base_url;

export function tablesUrl() {
  const url = BASE_URL + API.tables;
  return url;
}
