import React from "react";
import { SideNav } from "./SideNav";
import { useAppContext } from "../../context/App";
import { LAYOUT } from "../../utils/constants";
import { LayoutToolbar } from "../Toolbar";

export const Layout = ({ children }: { children: React.ReactNode }) => {
  const { toolbarMode } = useAppContext();
  return (
    <div className="layout">
      <SideNav />
      <LayoutToolbar />
      <div
        className="layout__content"
        id="layout__content"
        style={{
          paddingTop:
            toolbarMode === "TRAINING" ? LAYOUT.training_toolbar_height : 0,
        }}
      >
        {children}
      </div>
    </div>
  );
};
