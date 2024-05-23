import { ChakraProvider } from "@chakra-ui/provider";
import React from "react";
import ReactDOM from "react-dom/client";
import { ToastContainer } from "react-toastify";
import App from "./App";
import { AppProvider } from "./context";
import "./styles/css/styles.css";
import { customChakraTheme } from "./theme";

const root = ReactDOM.createRoot(
  document.getElementById("root") as HTMLElement,
);
root.render(
  <React.StrictMode>
    <ChakraProvider theme={customChakraTheme}>
      <AppProvider>
        <ToastContainer />
        <App />
      </AppProvider>
    </ChakraProvider>
  </React.StrictMode>,
);
