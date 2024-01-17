import { useEffect, useRef } from "react";

interface Props {
  eventAction: (event: KeyboardEvent) => void;
  addToDom?: boolean;
}

export const useKeyListener = ({ eventAction, addToDom = true }: Props) => {
  const eventActionRef = useRef(eventAction);
  useEffect(() => {
    eventActionRef.current = eventAction;
  }, [eventAction]);

  useEffect(() => {
    const handleEvent = (event: KeyboardEvent) => {
      eventActionRef.current(event);
    };
    if (addToDom) {
      window.addEventListener("keydown", handleEvent);
    }
    return () => {
      window.removeEventListener("keydown", handleEvent);
    };
  }, [addToDom]);
};
