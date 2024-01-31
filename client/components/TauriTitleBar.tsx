import React from "react";
import { useAppContext } from "../context/App";
import { FaGear } from "react-icons/fa6";
import { COLOR_DARK_BG_PRIMARY } from "../utils/colors";
import { HoverableIcon } from "./HoverableIcon";
import { PATHS } from "../utils/constants";
import { WindowTitlebar } from "../thirdparty/tauri-controls";
import { Spinner } from "@chakra-ui/react";

export const TauriTitleBar = () => {
  const { titleBarHeight } = useAppContext();
  const { platform, serverLaunched } = useAppContext();

  const getPlatform = () => {
    switch (platform) {
      case "macos":
        return "macos";
      case "windows":
        return "windows";

      case "linux":
        return "gnome";
    }
    return "macos";
  };
  return (
    <WindowTitlebar
      windowControlsProps={{
        platform: getPlatform(),
      }}
      controlsOrder={platform === "macos" ? "left" : "right"}
      style={{
        height: titleBarHeight,
        background: COLOR_DARK_BG_PRIMARY,
        borderTopRightRadius: "15px",
        borderTopLeftRadius: "15px",
        position: "fixed",
        width: "100%",
        zIndex: "1000000 !important",
        top: 0,
      }}
    >
      <div
        style={{
          height: titleBarHeight,
          display: "flex",
          alignItems: "center",
          justifyContent: "flex-end",
          width: "100%",
          paddingRight: "16px",
          gap: "16px",
        }}
        data-tauri-drag-region
      >
        {!serverLaunched && (
          <div>
            <Spinner size={"xs"} />
          </div>
        )}
        <HoverableIcon
          icon={(props) => <FaGear {...props} />}
          to={PATHS.settings}
        />
      </div>
    </WindowTitlebar>
  );
};
