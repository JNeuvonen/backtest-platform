import { useToast } from "@chakra-ui/react";
import React, {
  createContext,
  useState,
  ReactNode,
  useContext,
  useEffect,
  useRef,
  MutableRefObject,
} from "react";
import { URLS } from "../clients/endpoints";

interface LogContextType {
  logs: LogMessage[];
}

export const LogContext = createContext<LogContextType>({
  logs: [],
});

interface LogProviderProps {
  children: ReactNode;
}

interface LogMessage {
  msg: string;
  log_level: number;
  display: boolean;
  refetch: boolean;
  dom_event: string;
}

interface DispatchDomEventProps {
  channel: string;
  data?: string;
}

export const dispatchDomEvent = ({
  channel,
  data = "",
}: DispatchDomEventProps) => {
  const message = new CustomEvent(channel, { detail: data });
  window.dispatchEvent(message);
};

export const LogProvider: React.FC<LogProviderProps> = ({ children }) => {
  const [logs, setLogs] = useState<LogMessage[]>([]);
  const toast = useToast();
  const websocketRef: MutableRefObject<WebSocket | null> = useRef(null);

  const getToastStatus = (
    logLevel: number
  ): "info" | "warning" | "error" | "success" => {
    if (logLevel === 20) {
      return "info";
    }
    if (logLevel === 30) {
      return "warning";
    }

    if (logLevel === 40) {
      return "error";
    }
    return "success";
  };

  useEffect(() => {
    if (!websocketRef.current) {
      websocketRef.current = new WebSocket(URLS.ws_streams_log);
      const websocket = websocketRef.current;
      websocket.onmessage = (event) => {
        const data: LogMessage = JSON.parse(event.data);
        if (data.dom_event) {
          dispatchDomEvent({ channel: data.dom_event, data: data.msg });
        }
        if (data.display) {
          toast({
            title: data.msg,
            status: getToastStatus(data.log_level),
            duration: 5000,
            isClosable: true,
          });
        }
        addLog(data);
      };

      websocket.onerror = (error) => {
        console.error("WebSocket Error:", error);
      };
    }
  }, [toast]);

  const addLog = (newLog: LogMessage) => {
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
