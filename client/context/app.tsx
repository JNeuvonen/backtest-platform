import React, {
  createContext,
  useState,
  ReactNode,
  useContext,
  useEffect,
} from "react";
import { invoke } from "@tauri-apps/api/tauri";
import { LAYOUT, TAURI_COMMANDS } from "../utils/constants";

export type Platform = "" | "macos" | "windows" | "linux";

interface AppContextType {
  platform: Platform;
  contentIndentPx: number;
  innerSideNavWidth: number;
}

export const AppContext = createContext<AppContextType>({
  platform: "macos",
  contentIndentPx: LAYOUT.side_nav_width_px,
  innerSideNavWidth: LAYOUT.inner_side_nav_width_px,
});

interface AppProviderProps {
  children: ReactNode;
}

export const AppProvider: React.FC<AppProviderProps> = ({ children }) => {
  const [platform, setPlatform] = useState<Platform>("");
  const [contentIndentPx] = useState(LAYOUT.side_nav_width_px);
  const [layoutPaddingPx] = useState(LAYOUT.layout_padding_px);
  const [innerSideNavWidth] = useState(LAYOUT.inner_side_nav_width_px);

  useEffect(() => {
    const fetchAppData = async () => {
      try {
        const platform: Platform = await invoke(TAURI_COMMANDS.fetch_platform);
        setPlatform(platform);
      } catch (error) {
        console.error("Error fetching app data:", error);
      }
    };

    fetchAppData();
  }, []);

  return (
    <AppContext.Provider
      value={{
        platform,
        contentIndentPx: contentIndentPx + layoutPaddingPx,
        innerSideNavWidth,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};

export const useAppContext = () => useContext(AppContext);
