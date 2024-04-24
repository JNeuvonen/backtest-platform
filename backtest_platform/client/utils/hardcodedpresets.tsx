const genOptiontype = (pair: string) => {
  return {
    label: pair,
    value: pair,
  };
};

export const BULK_SIM_PAIR_PRESETS = [
  {
    label: "Speculative - weak altcoins",
    pairs: [
      genOptiontype("FARMUSDT"),
      genOptiontype("ENSUSDT"),
      genOptiontype("IOTAUSDT"),
      genOptiontype("FISUSDT"),
      genOptiontype("GNSUSDT"),
      genOptiontype("HARDUSDT"),
      genOptiontype("HOOKUSDT"),
      genOptiontype("HIGHUSDT"),
      genOptiontype("ICPUSDT"),
      genOptiontype("ICXUSDT"),
      genOptiontype("IDEXUSDT"),
      genOptiontype("IRISUSDT"),
      genOptiontype("JUVUSDT"),
      genOptiontype("KDAUSDT"),
      genOptiontype("KLAYUSDT"),
      genOptiontype("LAZIOUSDT"),
      genOptiontype("LITUSDT"),
      genOptiontype("LOKAUSDT"),
      genOptiontype("MBOXUSDT"),
      genOptiontype("MINAUSDT"),
    ],
  },
  {
    label: "Speculative - solid altcoins",
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
