import React from "react";
import { useAppContext } from "../context/App";
import { FaGear } from "react-icons/fa6";
import { COLOR_DARK_BG_PRIMARY } from "../utils/colors";
import { HoverableIcon } from "./HoverableIcon";
import { PATHS } from "../utils/constants";
import { WindowTitlebar } from "../thirdparty/tauri-controls";
import { IconButton, Spinner } from "@chakra-ui/react";
import { MdArrowForward } from "react-icons/md";
import { BUTTON_VARIANTS } from "../theme";

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
          justifyContent: "space-between",
          width: "100%",
          paddingRight: "16px",
          gap: "16px",
          zIndex: "1000000 !important",
        }}
        data-tauri-drag-region
      >
        <div
          style={{
            display: "flex",
            gap: "6px",
            alignItems: "center",
            marginLeft: "48px",
          }}
        >
          <IconButton
            aria-label="Icon button"
            icon={<MdArrowForward style={{ transform: "rotate(180deg)" }} />}
            variant={BUTTON_VARIANTS.grey}
            onClick={() => window.history.back()}
            height={"30px"}
            width={"30px"}
          />
          <IconButton
            aria-label="Icon button"
            icon={<MdArrowForward />}
            variant={BUTTON_VARIANTS.grey}
            onClick={() => window.history.forward()}
            height={"30px"}
            width={"30px"}
          />
        </div>

        <div
          style={{
            display: "flex",
            gap: "6px",
            alignItems: "center",
            marginRight: "16px",
          }}
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
      </div>
    </WindowTitlebar>
  );
};
