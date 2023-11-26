import { useState } from "react";

interface Props {
  messageChannel: string;
}

export const useSendMessage = ({ messageChannel }: Props) => {
  const [msgName] = useState(messageChannel);
  const sendMessage = ({ messageData = "" }: { messageData?: string }) => {
    const message = new CustomEvent(msgName, { detail: messageData });
    window.dispatchEvent(message);
  };
  return {
    sendMessage,
  };
};
