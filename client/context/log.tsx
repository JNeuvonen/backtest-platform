import { useToast } from "@chakra-ui/react";
import React, {
  createContext,
  useState,
  ReactNode,
  useContext,
  useEffect,
} from "react";
import { URLS } from "../clients/endpoints";
import { SIGNAL_OPEN_TRAINING_TOOLBAR } from "../utils/constants";

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
  notification_duration?: number;
}

interface DispatchDomEventProps {
  channel: string;
  data?: string;
}

let socket: WebSocket | null;

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
    const timerId = setTimeout(() => {
      socket = new WebSocket(URLS.ws_streams_log);
      socket.onmessage = (event) => {
        if (event.data === "health") return;
        const data: LogMessage = JSON.parse(event.data);

        if (data.msg === SIGNAL_OPEN_TRAINING_TOOLBAR) {
        }

        if (data.dom_event) {
          dispatchDomEvent({ channel: data.dom_event, data: data.msg });
        }
        if (data.display) {
          toast({
            title: data.msg,
            status: getToastStatus(data.log_level),
            duration: data.notification_duration ?? 5000,
            isClosable: true,
          });
        }
        addLog(data);
      };

      socket.onerror = (error) => {
        console.error("WebSocket Error:", error);
      };
    }, 1000);
    return () => clearTimeout(timerId);
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
