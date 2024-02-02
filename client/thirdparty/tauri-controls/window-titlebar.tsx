import React from "react";
import { useEffect, useState } from "react";
import { cn } from "./libs/utils";
import type { WindowTitlebarProps } from "./types";
import {
  WindowControls,
  convertPlatformToRightFormat,
} from "./window-controls";
import { useAppContext } from "../../context/app";

export function WindowTitlebar({
  children,
  controlsOrder = "system",
  className,
  windowControlsProps,
  ...props
}: WindowTitlebarProps) {
  const [osType, setOsType] = useState<string | undefined>(undefined);
  const { platform: platformFromEnv } = useAppContext();

  useEffect(() => {
    setOsType(convertPlatformToRightFormat(platformFromEnv));
  }, [platformFromEnv]);

  const left =
    controlsOrder === "left" ||
    (controlsOrder === "platform" &&
      windowControlsProps?.platform === "macos") ||
    (controlsOrder === "system" && osType === "macos");

  const customProps = (ml: string) => {
    if (windowControlsProps?.justify !== undefined) return windowControlsProps;

    const { className: windowControlsClassName, ...restProps } =
      windowControlsProps || {};
    return {
      justify: false,
      className: cn(windowControlsClassName, ml),
      ...restProps,
    };
  };

  return (
    <div
      className={cn(
        "bg-background flex select-none flex-row overflow-hidden",
        className
      )}
      data-tauri-drag-region
      {...props}
      style={{ ...props.style, zIndex: 100000 }}
    >
      {left ? (
        <>
          <WindowControls {...customProps("ml-0")} />
          {children}
        </>
      ) : (
        <>
          {children}
          <WindowControls {...customProps("ml-auto")} />
        </>
      )}
    </div>
  );
}
