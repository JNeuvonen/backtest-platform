import React from "react";
import { useEffect, useState } from "react";
import { TauriAppWindowProvider } from "./contexts/plugin-window";
import { Gnome, MacOS, Windows } from "./controls";
import type { WindowControlsProps } from "./types";
import { useAppContext } from "../../context/app";
import { cn } from "./libs/utils";

export const convertPlatformToRightFormat = (platform: string) => {
  switch (platform) {
    case "macos":
      return "macos";
    case "linux":
      return "gnome";
    case "windows":
      return "windows";
  }
  return "macos";
};

export function WindowControls({
  platform,
  justify = false,
  hide = false,
  hideMethod = "display",
  className,
  ...props
}: WindowControlsProps) {
  const [osType, setOsType] = useState<string | undefined>(undefined);
  const { platform: platformFromEnv } = useAppContext();

  useEffect(() => {
    setOsType(convertPlatformToRightFormat(platformFromEnv));
  }, [platform]);

  const customClass = cn(
    "flex",
    className,
    hide && (hideMethod === "display" ? "hidden" : "invisible")
  );

  // Determine the default platform based on the operating system if not specified
  if (!platform) {
    switch (osType) {
      case "macos":
        platform = "macos";
        break;
      case "linux":
        platform = "gnome";
        break;
      default:
        platform = "windows";
    }
  }

  const ControlsComponent = () => {
    switch (platform) {
      case "windows":
        return (
          <Windows
            className={cn(customClass, justify && "ml-auto")}
            {...props}
          />
        );
      case "macos":
        return (
          <MacOS className={cn(customClass, justify && "ml-0")} {...props} />
        );
      case "gnome":
        return (
          <Gnome className={cn(customClass, justify && "ml-auto")} {...props} />
        );
      default:
        return (
          <Windows
            className={cn(customClass, justify && "ml-auto")}
            {...props}
          />
        );
    }
  };

  return (
    <TauriAppWindowProvider>
      <ControlsComponent />
    </TauriAppWindowProvider>
  );
}
