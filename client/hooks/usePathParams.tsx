import { useParams } from "react-router-dom";

interface Props {
  key: string;
}

type PathParams = {
  [key: string]: string;
};

export const usePathParams = ({ key }: Props) => {
  const params = useParams<PathParams>();
  const item = params[key.replace(":", "")];
  return item ?? "";
};
