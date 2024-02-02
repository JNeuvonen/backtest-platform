# Why this projects exists?

I wanted to experiment with what kind of cool stuff you can build with a desktop application that has a Python environment packaged into it. That allows for building an application with a UI that can execute Python locally while not being dependent on the system's Python. 

The setup goes as follows:

- Packaging Python runtime and projects Python files ([pyserver](https://github.com/JNeuvonen/backtest-engine/tree/master/pyserver)) and dependencies to a binary using [pyoxidizer](https://github.com/indygreg/PyOxidizer)
- Embedding that Python binary into a [tauri](https://github.com/tauri-apps/tauri) desktop application
- Starting the Python binary (FastAPI web server) from within Tauri's rust backend ([main.rs](https://github.com/JNeuvonen/backtest-engine/blob/master/src-tauri/src/main.rs#L64-L69)).
- Now the [frontend](https://github.com/JNeuvonen/backtest-engine/tree/master/client) (Chromium powered) can call the Python server running locally without being dependent on or interfering with the system's Python.

So basically, Tauri is only used to bootstrap the local Python environment.
