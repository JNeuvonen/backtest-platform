import React from "react";
import { useAppContext } from "../context/App";
import { FaGear } from "react-icons/fa6";
import { COLOR_DARK_BG_PRIMARY } from "../utils/colors";
import { HoverableIcon } from "./HoverableIcon";
import { PATHS } from "../utils/constants";
import { WindowTitlebar } from "../thirdparty/tauri-controls";

const MacOsTitleBar = () => {
  const { titleBarHeight } = useAppContext();
  const { platform } = useAppContext();

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
        }}
        data-tauri-drag-region
      >
        <HoverableIcon
          icon={(props) => <FaGear {...props} />}
          to={PATHS.settings}
        />
      </div>
    </WindowTitlebar>
  );
};

const LinuxTitleBar = () => {
  const { titleBarHeight } = useAppContext();
  return (
    <WindowTitlebar
      windowControlsProps={{ platform: "gnome" }}
      controlsOrder="right"
      style={{ height: titleBarHeight }}
    >
      <div
        style={{
          height: titleBarHeight,
          display: "flex",
          alignItems: "center",
        }}
      >
        Test from linux
      </div>
    </WindowTitlebar>
  );
};

const WindowsTitleBar = () => {
  const { titleBarHeight } = useAppContext();
  return (
    <WindowTitlebar
      windowControlsProps={{ platform: "gnome" }}
      controlsOrder="right"
      style={{ height: titleBarHeight }}
    >
      <div
        style={{
          height: titleBarHeight,
          display: "flex",
          alignItems: "center",
        }}
      >
        Test from windows
      </div>
    </WindowTitlebar>
  );
};

export const TauriTitleBar = () => {
  const { platform } = useAppContext();

  if (platform === "macos") {
    return <MacOsTitleBar />;
  }

  if (platform === "windows") {
    return <WindowsTitleBar />;
  }

  if (platform === "linux") {
    return <LinuxTitleBar />;
  }

  return <div>Title bar TODO for other platforms than mac</div>;
};
