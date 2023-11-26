import { useToast } from "@chakra-ui/react";
import React, {
  createContext,
  useState,
  ReactNode,
  useContext,
  useEffect,
  useRef,
} from "react";
import { STREAMS_LOG, URLS } from "../clients/endpoints";

type LogMessage = string;

interface LogContextType {
  logs: LogMessage[];
}

export const LogContext = createContext<LogContextType>({
  logs: [],
});

interface LogProviderProps {
  children: ReactNode;
}

export const LogProvider: React.FC<LogProviderProps> = ({ children }) => {
  const [logs, setLogs] = useState<LogMessage[]>([]);
  const toast = useToast();
  const websocketRef = useRef(null);

  let websocket: null | WebSocket = null;

  useEffect(() => {
    if (!websocketRef.current) {
      websocket = new WebSocket(URLS.ws_streams_log);
      websocket.onmessage = (event) => {
        const message = event.data;
        addLog(message);
        if (message.startsWith(STREAMS_LOG.error)) {
        } else if (message.startsWith(STREAMS_LOG.warning)) {
        } else if (message.startsWith(STREAMS_LOG.info)) {
        } else if (message.startsWith(STREAMS_LOG.debug)) {
        }
      };

      websocket.onerror = (error) => {
        console.error("WebSocket Error:", error);
      };
    }
  }, []);

  const addLog = (newLog: LogMessage) => {
    console.log(newLog);
    setLogs((prevLogs) => {
      const updatedLogs = [newLog, ...prevLogs];
      if (updatedLogs.length > 1000) {
        return updatedLogs.slice(0, 1000);
      }
      return updatedLogs;
    });
  };

  return <LogContext.Provider value={{ logs }}>{children}</LogContext.Provider>;
};

export const useLogs = () => useContext(LogContext);
