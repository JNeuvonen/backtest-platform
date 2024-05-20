import {
  createContext,
  ReactNode,
  useContext,
  useEffect,
  useState,
} from "react";
import { useWinDimensions } from "src/hooks";
import { MOBILE_WIDTH_CUTOFF } from "src/utils";

interface AppContextType {
  isMobileLayout: boolean;
  setIsMobileLayout: React.Dispatch<React.SetStateAction<boolean>>;
}

interface AppProviderProps {
  children: ReactNode;
}

export const AppContext = createContext<AppContextType>({} as AppContextType);

export const AppProvider: React.FC<AppProviderProps> = ({ children }) => {
  const { width } = useWinDimensions();
  const [isMobileLayout, setIsMobileLayout] = useState(
    width < MOBILE_WIDTH_CUTOFF,
  );

  useEffect(() => {
    if (width < MOBILE_WIDTH_CUTOFF) {
      setIsMobileLayout(true);
    } else {
      setIsMobileLayout(false);
    }
  }, [width]);

  return (
    <AppContext.Provider
      value={{
        isMobileLayout,
        setIsMobileLayout,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};

export const useAppContext = () => useContext(AppContext);
