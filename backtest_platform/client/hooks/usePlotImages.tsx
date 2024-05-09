import { useEffect, useState } from "react";

interface Props {
  apiQuery: object;
  key: string;
}

export const usePlotImage = ({ apiQuery, key }: Props) => {
  const [image, setImage] = useState("");

  useEffect(() => {
    if (apiQuery) {
      setImage(`data:iamge/png;base64,${apiQuery[key]}`);
    }
  }, [apiQuery]);
  return image;
};
