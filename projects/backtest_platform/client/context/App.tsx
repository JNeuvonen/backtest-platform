import React, {
  createContext,
  useState,
  ReactNode,
  useContext,
  useEffect,
} from "react";

import { invoke } from "@tauri-apps/api/tauri";
import { LAYOUT, TAURI_COMMANDS } from "../utils/constants";
import { LOCAL_API_URI } from "../clients/endpoints";
import { DISK_KEYS, DiskManager } from "../utils/disk";
import { createPredServApiKey } from "../clients/requests";

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
  trainJobId: string;
  setInnerSideNavWidth: React.Dispatch<React.SetStateAction<number>>;
  titleBarHeight: number;
  serverLaunched: boolean;
  appSettings: AppSettings | null;
  setAppSettings: React.Dispatch<React.SetStateAction<AppSettings | null>>;
  updatePredServAPIKey: (newApiKey: string) => void;
  getPredServAPIKey: () => string;
}

export const AppContext = createContext<AppContextType>({} as AppContextType);

interface AppProviderProps {
  children: ReactNode;
}

interface AppSettings {
  predServAPIKey: string;
}

const settingsDiskManager = new DiskManager(DISK_KEYS.app_settings);

export const AppProvider: React.FC<AppProviderProps> = ({ children }) => {
  const [platform, setPlatform] = useState<Platform>("");
  const [contentIndentPx] = useState(LAYOUT.side_nav_width_px);
  const [layoutPaddingPx] = useState(LAYOUT.layout_padding_px);
  const [innerSideNavWidth, setInnerSideNavWidth] = useState(0);
  const [toolbarMode, setToolbarMode] = useState<ToolbarMode>("");
  const [epochsRan, setEpochsRan] = useState(0);
  const [maximumEpochs, setMaximumEpochs] = useState(0);
  const [trainLosses, setTrainLosses] = useState<number[]>([]);
  const [valLosses, setValLosses] = useState<number[]>([]);
  const [epochTime, setEpochTime] = useState<number>(0);
  const [trainJobId, setTrainJobId] = useState("");
  const [titleBarHeight] = useState(40);
  const [serverLaunched, setServerLaunched] = useState(false);
  const [appSettings, setAppSettings] = useState<AppSettings | null>(
    settingsDiskManager.read()
  );

  useEffect(() => {
    fetch(LOCAL_API_URI + "/")
      .then((response) => {
        if (response.status === 200) {
          setServerLaunched(true);
        }
        return response.json();
      })
      .catch();
  }, []);

  useEffect(() => {
    const fetchAppData = async () => {
      try {
        const platform: Platform = await invoke(TAURI_COMMANDS.fetch_platform);
        setPlatform(platform);
      } catch (error) {
        console.error("Error fetching app data:", error);
      }
    };

    const fetchAppSettings = async () => {
      if (appSettings === null) {
        const apiKey = await createPredServApiKey();
        const settingsDict = {
          predServAPIKey: apiKey,
        };
        settingsDiskManager.save(settingsDict);
      }
    };

    fetchAppData();
    fetchAppSettings();
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

  const updatePredServAPIKey = (newApiKey: string) => {
    const settingsDict = {
      predServAPIKey: newApiKey,
    };
    settingsDiskManager.save(settingsDict);
    setAppSettings(settingsDict);
  };

  const getPredServAPIKey = () => {
    return appSettings?.predServAPIKey || "";
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
        trainJobId,
        setInnerSideNavWidth,
        titleBarHeight,
        serverLaunched,
        appSettings,
        setAppSettings,
        updatePredServAPIKey,
        getPredServAPIKey,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};

export const useAppContext = () => useContext(AppContext);
