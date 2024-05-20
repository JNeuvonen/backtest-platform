import { useEffect, useRef } from "react";

interface Props {
  isEnabled: boolean;
  onClickCallback: () => void;
}

export const PAGE_BLUR_ID = "blur";

export const Blur = ({ isEnabled, onClickCallback }: Props) => {
  const blurRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    blurRef.current?.addEventListener("click", () => {
      onClickCallback();
    });
  }, []);
  return (
    <div
      style={{
        position: "fixed",
        left: 0,
        right: 0,
        bottom: 0,
        top: 0,
        zIndex: 1000,
        display: isEnabled ? "flex" : "none",
        backgroundColor: "rgba(0, 0, 0, 0.75)",
      }}
      ref={blurRef}
      id={PAGE_BLUR_ID}
    />
  );
};
