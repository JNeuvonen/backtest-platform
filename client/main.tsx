import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { customChakraTheme } from "./theme";
import { ChakraProvider } from "@chakra-ui/react";
import App from "./App";
import "./styles/css/styles.css";
import { LogProvider } from "./context/log";
import { AppProvider } from "./context/app";
import "monaco-editor/min/vs/editor/editor.main.css";
import { appWindow } from "@tauri-apps/api/window";
import { CONSTANTS } from "./utils/constants";

const queryClient = new QueryClient();

appWindow.onCloseRequested(() => {
  fetch(CONSTANTS.base_url + "/shutdown", {
    method: "POST",
  });
});

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <AppProvider>
      <QueryClientProvider client={queryClient}>
        <ChakraProvider theme={customChakraTheme}>
          <LogProvider>
            <App />
          </LogProvider>
        </ChakraProvider>
      </QueryClientProvider>
    </AppProvider>
  </React.StrictMode>
);
