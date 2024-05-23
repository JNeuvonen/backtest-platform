import { useAuth0 } from "@auth0/auth0-react";

export const LoginPage = () => {
  const { loginWithRedirect } = useAuth0();
  return (
    <div className="layout__content" style={{ marginLeft: "0px" }}>
      <button onClick={() => loginWithRedirect()}>Log In</button>
    </div>
  );
};
