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
}

export const AppContext = createContext<AppContextType>({
  platform: "macos",
  contentIndentPx: LAYOUT.side_nav_width,
});

interface AppProviderProps {
  children: ReactNode;
}

export const AppProvider: React.FC<AppProviderProps> = ({ children }) => {
  const [platform, setPlatform] = useState<Platform>("");
  const [contentIndentPx] = useState(LAYOUT.side_nav_width);
  const [layoutPaddingPx] = useState(LAYOUT.layout_padding);

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
      value={{ platform, contentIndentPx: contentIndentPx + layoutPaddingPx }}
    >
      {children}
    </AppContext.Provider>
  );
};

export const useAppContext = () => useContext(AppContext);
