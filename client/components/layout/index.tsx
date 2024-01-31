import React from "react";
import { SideNav } from "./SideNav";
import { useAppContext } from "../../context/App";
import { LAYOUT } from "../../utils/constants";
import { LayoutToolbar } from "../Toolbar";
import { TauriTitleBar } from "../TauriTitleBar";

export const Layout = ({ children }: { children: React.ReactNode }) => {
  const { toolbarMode } = useAppContext();
  const { titleBarHeight } = useAppContext();
  return (
    <div
      className="layout"
      style={{
        borderRadius: "15px !important",
      }}
    >
      <TauriTitleBar />
      <SideNav />
      <LayoutToolbar />
      <div
        className="layout__content"
        id="layout__content"
        style={{
          marginTop:
            toolbarMode === "TRAINING"
              ? LAYOUT.training_toolbar_height + titleBarHeight
              : titleBarHeight,
        }}
      >
        {children}
      </div>
    </div>
  );
};
