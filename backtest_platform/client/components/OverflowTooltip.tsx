import React, {
  useState,
  useEffect,
  useRef,
  cloneElement,
  Children,
} from "react";
import { Tooltip } from "@chakra-ui/react";

interface Props {
  text: string;
  children: React.ReactElement;
  containerId: string;
}

export const OverflopTooltip = ({ text, children, containerId }: Props) => {
  const [isOverflowing, setIsOverflowing] = useState(false);
  const childRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const checkOverflow = () => {
      const container = document.getElementById(containerId);
      const child = childRef.current;

      if (child && container) {
        const isChildOverflowing = child.scrollWidth > child.clientWidth;
        setIsOverflowing(isChildOverflowing);
      }
    };

    checkOverflow();

    const resizeObserver = new ResizeObserver(checkOverflow);
    if (childRef.current) {
      resizeObserver.observe(childRef.current);
    }

    return () => {
      resizeObserver.disconnect();
    };
  }, [containerId]);

  return (
    <Tooltip label={text} isDisabled={!isOverflowing}>
      {cloneElement(Children.only(children), {
        style: {
          whiteSpace: "nowrap",
          overflow: "hidden",
          textOverflow: "ellipsis",
          maxWidth: "100%",
        },
        ref: childRef,
      })}
    </Tooltip>
  );
};
