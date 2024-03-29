import { useEffect } from "react";

interface Props {
  messageName: string;
  messageCallback: () => void;
}

export const useMessageListener = ({ messageName, messageCallback }: Props) => {
  useEffect(() => {
    window.addEventListener(messageName, messageCallback);
    return () => window.removeEventListener(messageName, messageCallback);
  }, [messageCallback, messageName]);
};
