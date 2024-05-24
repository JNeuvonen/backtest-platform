import React from "react";
import { useAuth0 } from "@auth0/auth0-react";
import {
  createContext,
  ReactNode,
  useContext,
  useEffect,
  useState,
} from "react";
import { fetchUserParams, httpReq } from "../http";
import { LoginPage } from "src/pages/login";

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
  const { isAuthenticated, user } = useAuth0();
  const [userFromDb, setUserFromDb] = useState<any>(null);

  useEffect(() => {
    const asyncHelper = async () => {
      const userRes = await httpReq({ ...fetchUserParams() });
      if (userRes && userRes.success) {
        setUserFromDb(userRes);
      }
    };

    asyncHelper();
  }, [isAuthenticated, user]);

  if (!userFromDb) {
    return <LoginPage />;
  }

  return (
    <UserContext.Provider value={{ user: null }}>
      {children}
    </UserContext.Provider>
  );
};

export const useUserContext = () => useContext(UserContext);
