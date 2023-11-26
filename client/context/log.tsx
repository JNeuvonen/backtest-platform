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
import { STREAMS_LOG, URLS } from "../clients/endpoints";
import { useSendMessage } from "../hooks/useSendMessage";
import { DOM_MESSAGES } from "../utils/constants";

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
  const websocketRef: MutableRefObject<WebSocket | null> = useRef(null);
  const { sendMessage } = useSendMessage({
    messageChannel: DOM_MESSAGES.refetch,
  });

  useEffect(() => {
    if (!websocketRef.current) {
      websocketRef.current = new WebSocket(URLS.ws_streams_log);
      const websocket = websocketRef.current;
      websocket.onmessage = (event) => {
        let message = event.data;

        let messageParts = message.split(":");
        let msgMetadata = messageParts[0];
        let msgData = messageParts[1];

        const shouldRefetch = msgMetadata.includes(
          STREAMS_LOG.UTILS.should_refetch
        );
        addLog(message);

        if (shouldRefetch) {
          sendMessage({});
        }

        if (msgMetadata.includes(STREAMS_LOG.error)) {
        } else if (msgMetadata.includes(STREAMS_LOG.warning)) {
        } else if (msgMetadata.includes(STREAMS_LOG.info)) {
          console.log("Executing here");
          message = message.replace(STREAMS_LOG.info, "");
          toast({
            title: msgData,
            status: "success",
            duration: 5000,
            isClosable: true,
          });
        } else if (msgMetadata.includes(STREAMS_LOG.debug)) {
        }
      };

      websocket.onerror = (error) => {
        console.error("WebSocket Error:", error);
      };
    }
  }, []);

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
