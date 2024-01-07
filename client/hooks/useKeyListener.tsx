import { useEffect } from "react";

interface Props {
  eventAction: (event: KeyboardEvent) => void;
  addToDom?: boolean;
}

export const useKeyListener = ({ eventAction, addToDom = true }: Props) => {
  useEffect(() => {
    if (addToDom) {
      window.addEventListener("keydown", eventAction);
    }
    return () => {
      window.removeEventListener("keydown", eventAction);
    };
  }, [addToDom, eventAction]);
};
