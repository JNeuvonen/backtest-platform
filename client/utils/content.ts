import { Platform } from "../context/app";

const OS_NOT_RECOGNIZED_MSG = "OS was not recognized";

export const KEYBIND_MSGS = {
  get_save: (platform: Platform) => {
    if (platform === "macos") {
      return "Press [CMD + s] to save";
    }
    if (platform === "linux" || platform === "windows") {
      return "Press [CTRL + s] to save";
    }
    return OS_NOT_RECOGNIZED_MSG;
  },
};
