const genOptiontype = (pair: string) => {
  return {
    label: pair,
    value: pair,
  };
};

export const BULK_SIM_PAIR_PRESETS = [
  {
    label: "Speculative coins",
    pairs: [
      genOptiontype("ALGOUSDT"),
      genOptiontype("GALAUSDT"),
      genOptiontype("IOTAUSDT"),
      genOptiontype("FLOWUSDT"),
      genOptiontype("AXSUSDT"),
      genOptiontype("SANDUSDT"),
      genOptiontype("TRXUSDT"),
      genOptiontype("CVCUSDT"),
      genOptiontype("DEGOUSDT"),
      genOptiontype("DOCKUSDT"),
      genOptiontype("DUSKUSDT"),
      genOptiontype("EGLDUSDT"),
      genOptiontype("ELFUSDT"),
      genOptiontype("ETCUSDT"),
      genOptiontype("FETUSDT"),
      genOptiontype("FIOUSDT"),
      genOptiontype("FLOKIUSDT"),
      genOptiontype("FUNUSDT"),
      genOptiontype("GASUSDT"),
    ],
  },
];
