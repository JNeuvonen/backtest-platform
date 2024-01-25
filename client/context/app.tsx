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
export type ToolbarMode = "TRAINING" | "";

interface AppContextType {
  platform: Platform;
  contentIndentPx: number;
  innerSideNavWidth: number;
  openTrainingToolbar: () => void;
  parseEpochMessage: (msg: string) => void;
  closeToolbar: () => void;
  toolbarMode: ToolbarMode;
  epochsRan: number;
  maximumEpochs: number;
  trainLosses: number[];
  valLosses: number[];
  epochTime: number;
}

export const AppContext = createContext<AppContextType>({} as AppContextType);

interface AppProviderProps {
  children: ReactNode;
}

export const AppProvider: React.FC<AppProviderProps> = ({ children }) => {
  const [platform, setPlatform] = useState<Platform>("");
  const [contentIndentPx] = useState(LAYOUT.side_nav_width_px);
  const [layoutPaddingPx] = useState(LAYOUT.layout_padding_px);
  const [innerSideNavWidth] = useState(LAYOUT.inner_side_nav_width_px);
  const [toolbarMode, setToolbarMode] = useState<ToolbarMode>("");
  const [epochsRan, setEpochsRan] = useState(0);
  const [maximumEpochs, setMaximumEpochs] = useState(0);
  const [trainLosses, setTrainLosses] = useState<number[]>([]);
  const [valLosses, setValLosses] = useState<number[]>([]);
  const [epochTime, setEpochTime] = useState<number>(0);
  const [trainJobId, setTrainJobId] = useState("");

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

  const openTrainingToolbar = () => {
    setToolbarMode("TRAINING");
  };

  const closeToolbar = () => {
    setToolbarMode("");

    setEpochsRan(0);
    setEpochTime(0);
    setMaximumEpochs(0);
    setTrainLosses([]);
    setValLosses([]);
  };

  const parseEpochMessage = (msg: string) => {
    try {
      const data = msg.split("\n")[1].split("/");

      const epochsRan = data[0];
      const maxEpochs = data[1];
      const trainLoss = data[2];
      const valLoss = data[3];
      const epochTime = data[4];
      const trainJobId = data[5];

      setEpochsRan(Number(epochsRan));
      setMaximumEpochs(Number(maxEpochs));
      setTrainLosses((prevTrainLosses) => [
        ...prevTrainLosses,
        Number(trainLoss),
      ]);
      setValLosses((prevValLosses) => [...prevValLosses, Number(valLoss)]);
      setEpochTime(Number(epochTime));
      setTrainJobId(trainJobId);
    } catch {}
  };

  return (
    <AppContext.Provider
      value={{
        platform,
        contentIndentPx: contentIndentPx + layoutPaddingPx,
        innerSideNavWidth,
        openTrainingToolbar,
        closeToolbar,
        toolbarMode,
        parseEpochMessage,
        epochsRan,
        maximumEpochs,
        trainLosses,
        valLosses,
        epochTime,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};

export const useAppContext = () => useContext(AppContext);
