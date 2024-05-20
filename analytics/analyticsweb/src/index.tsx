import { ChakraProvider } from "@chakra-ui/provider";
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./styles/css/styles.css";
import { customChakraTheme } from "./theme";

const root = ReactDOM.createRoot(
  document.getElementById("root") as HTMLElement,
);
root.render(
  <React.StrictMode>
    <ChakraProvider theme={customChakraTheme}>
      <App />
    </ChakraProvider>
  </React.StrictMode>,
);
