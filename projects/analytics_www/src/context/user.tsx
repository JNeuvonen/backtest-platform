import React from "react";
import { useAuth0 } from "@auth0/auth0-react";
import {
  createContext,
  ReactNode,
  useContext,
  useEffect,
  useState,
} from "react";
import { fetchUserParams } from "../http";
import { useAccessTokenReq } from "src/hooks/useAccessTokenReq";
import { useToast } from "@chakra-ui/react";

interface UserContextType {
  user: any;
}

interface UserProviderProps {
  children: ReactNode;
}

export const UserContext = createContext<UserContextType>(
  {} as UserContextType,
);

export const UserProvider: React.FC<UserProviderProps> = ({ children }) => {
  const { isAuthenticated, user, getAccessTokenSilently } = useAuth0();
  const [userFromDb, setUserFromDb] = useState(null);
  const accessTokenReq = useAccessTokenReq();
  const toast = useToast();

  useEffect(() => {
    const asyncHelper = async () => {
      const token = await getAccessTokenSilently({
        authorizationParams: {
          audience: `https://${process.env.REACT_APP_AUTH0_DOMAIN}/api/v2/`,
          scope: "read:current_user",
        },
      });
      const user = await accessTokenReq({ ...fetchUserParams() });
      console.log(user);
    };

    asyncHelper();

    toast({
      title: "Title",
      status: "info",
      duration: 5000,
      isClosable: true,
    });
  }, [isAuthenticated, user]);

  return (
    <UserContext.Provider value={{ user: null }}>
      {children}
    </UserContext.Provider>
  );
};

export const useUserContext = () => useContext(UserContext);
