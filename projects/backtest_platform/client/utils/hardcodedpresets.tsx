const genOptiontype = (pair: string) => {
  return {
    label: pair,
    value: pair,
  };
};

export const BULK_SIM_PAIR_PRESETS = [
  {
    label: "Top 50",
    pairs: [
      genOptiontype("BTCUSDT"),
      genOptiontype("ETHUSDT"),
      genOptiontype("BNBUSDT"),
      genOptiontype("SOLUSDT"),
      genOptiontype("XRPUSDT"),
      genOptiontype("DOGEUSDT"),
      genOptiontype("ADAUSDT"),
      genOptiontype("SHIBUSDT"),
      genOptiontype("TRXUSDT"),
      genOptiontype("BCHUSDT"),
      genOptiontype("LINKUSDT"),
      genOptiontype("NEARUSDT"),
      genOptiontype("MATICUSDT"),
      genOptiontype("FETUSDT"),
      genOptiontype("ICPUSDT"),
      genOptiontype("UNIUSDT"),
      genOptiontype("RNDRUSDT"),
      genOptiontype("ETCUSDT"),
      genOptiontype("HBARUSDT"),
      genOptiontype("PEPEUSDT"),
      genOptiontype("ATOMUSDT"),
      genOptiontype("IMXUSDT"),
      genOptiontype("FILUSDT"),
    ],
  },
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
  {
    label: "Altcoins wide",
    pairs: [
      genOptiontype("ALGOUSDT"),
      genOptiontype("GALAUSDT"),
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
      genOptiontype("XRPUSDT"),
      genOptiontype("UTKUSDT"),
      genOptiontype("VICUSDT"),
      genOptiontype("VGXUSDT"),
      genOptiontype("TROYUSDT"),
      genOptiontype("UNFIUSDT"),
      genOptiontype("THETAUSDT"),
      genOptiontype("TFUELUSDT"),
      genOptiontype("TAOUSDT"),
      genOptiontype("AERGOUSDT"),
      genOptiontype("ACMUSDT"),
      genOptiontype("AGIXUSDT"),
      genOptiontype("AIUSDT"),
      genOptiontype("ALICEUSDT"),
      genOptiontype("ALPACAUSDT"),
      genOptiontype("ALPINEUSDT"),
      genOptiontype("ALTUSDT"),
      genOptiontype("ANKRUSDT"),
      genOptiontype("APEUSDT"),
      genOptiontype("APTUSDT"),
      genOptiontype("BONKUSDT"),
      genOptiontype("CITYUSDT"),
      genOptiontype("CTSIUSDT"),
      genOptiontype("CVXUSDT"),
      genOptiontype("DARUSDT"),
      genOptiontype("DATAUSDT"),
      genOptiontype("DCRUSDT"),
      genOptiontype("DENTUSDT"),
      genOptiontype("DEXEUSDT"),
      genOptiontype("DFUSDT"),
      genOptiontype("DGBUSDT"),
      genOptiontype("DIAUSDT"),
      genOptiontype("EDUUSDT"),
      genOptiontype("ENAUSDT"),
      genOptiontype("EPXUSDT"),
      genOptiontype("ERNUSDT"),
      genOptiontype("ETHFIUSDT"),
      genOptiontype("FIROUSDT"),
      genOptiontype("FLMUSDT"),
      genOptiontype("FLUXUSDT"),
      genOptiontype("FORTHUSDT"),
      genOptiontype("FORUSDT"),
      genOptiontype("FXSUSDT"),
      genOptiontype("GALUSDT"),
      genOptiontype("GFTUSDT"),
      genOptiontype("GHSTUSDT"),
      genOptiontype("GLMRUSDT"),
      genOptiontype("GLMUSDT"),
      genOptiontype("GMTUSDT"),
      genOptiontype("GMXUSDT"),
      genOptiontype("GNOUSDT"),
      genOptiontype("GRTUSDT"),
      genOptiontype("JSTUSDT"),
    ],
  },
  {
    label: "Altcoins +200",
    pairs: [
      genOptiontype("ALGOUSDT"),
      genOptiontype("GALAUSDT"),
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
      genOptiontype("XRPUSDT"),
      genOptiontype("UTKUSDT"),
      genOptiontype("VICUSDT"),
      genOptiontype("VGXUSDT"),
      genOptiontype("TROYUSDT"),
      genOptiontype("UNFIUSDT"),
      genOptiontype("THETAUSDT"),
      genOptiontype("TFUELUSDT"),
      genOptiontype("TAOUSDT"),
      genOptiontype("AERGOUSDT"),
      genOptiontype("ACMUSDT"),
      genOptiontype("AGIXUSDT"),
      genOptiontype("AIUSDT"),
      genOptiontype("ALICEUSDT"),
      genOptiontype("ALPACAUSDT"),
      genOptiontype("ALPINEUSDT"),
      genOptiontype("ALTUSDT"),
      genOptiontype("ANKRUSDT"),
      genOptiontype("APEUSDT"),
      genOptiontype("APTUSDT"),
      genOptiontype("BONKUSDT"),
      genOptiontype("CITYUSDT"),
      genOptiontype("CTSIUSDT"),
      genOptiontype("CVXUSDT"),
      genOptiontype("DARUSDT"),
      genOptiontype("DATAUSDT"),
      genOptiontype("DCRUSDT"),
      genOptiontype("DENTUSDT"),
      genOptiontype("DEXEUSDT"),
      genOptiontype("DFUSDT"),
      genOptiontype("DGBUSDT"),
      genOptiontype("DIAUSDT"),
      genOptiontype("EDUUSDT"),
      genOptiontype("ENAUSDT"),
      genOptiontype("EPXUSDT"),
      genOptiontype("ERNUSDT"),
      genOptiontype("ETHFIUSDT"),
      genOptiontype("FIROUSDT"),
      genOptiontype("FLMUSDT"),
      genOptiontype("FLUXUSDT"),
      genOptiontype("FORTHUSDT"),
      genOptiontype("FORUSDT"),
      genOptiontype("FXSUSDT"),
      genOptiontype("GALUSDT"),
      genOptiontype("GFTUSDT"),
      genOptiontype("GHSTUSDT"),
      genOptiontype("GLMRUSDT"),
      genOptiontype("GLMUSDT"),
      genOptiontype("GMTUSDT"),
      genOptiontype("GMXUSDT"),
      genOptiontype("GNOUSDT"),
      genOptiontype("GRTUSDT"),
      genOptiontype("JSTUSDT"),
      genOptiontype("BARUSDT"),
      genOptiontype("BEAMXUSDT"),
      genOptiontype("BELUSDT"),
      genOptiontype("BETAUSDT"),
      genOptiontype("BICOUSDT"),
      genOptiontype("BIFIUSDT"),
      genOptiontype("BLZUSDT"),
      genOptiontype("BLURUSDT"),
      genOptiontype("BNTUSDT"),
      genOptiontype("BNXUSDT"),
      genOptiontype("BOMEUSDT"),
      genOptiontype("BONDUSDT"),
      genOptiontype("BSWUSDT"),
      genOptiontype("BTTCUSDT"),
      genOptiontype("BURGERUSDT"),
      genOptiontype("C98USDT"),
      genOptiontype("CAKEUSDT"),
      genOptiontype("CELOUSDT"),
      genOptiontype("CELRUSDT"),
      genOptiontype("CFXUSDT"),
      genOptiontype("CHESSUSDT"),
      genOptiontype("CHRUSDT"),
      genOptiontype("CHZUSDT"),
      genOptiontype("CKBUSDT"),
      genOptiontype("CLVUSDT"),
      genOptiontype("COMBOUSDT"),
      genOptiontype("COMPUSDT"),
      genOptiontype("COSUSDT"),
      genOptiontype("COTIUSDT"),
      genOptiontype("CREAMUSDT"),
      genOptiontype("CRVUSDT"),
      genOptiontype("CTKUSDT"),
      genOptiontype("CTXCUSDT"),
      genOptiontype("CVPUSDT"),
      genOptiontype("CYBERUSDT"),
      genOptiontype("DASHUSDT"),
      genOptiontype("DODOUSDT"),
      genOptiontype("DOTUSDT"),
      genOptiontype("DYDXUSDT"),
      genOptiontype("DYMUSDT"),
      genOptiontype("ENJUSDT"),
      genOptiontype("EOSUSDT"),
      genOptiontype("FIDAUSDT"),
      genOptiontype("FILUSDT"),
      genOptiontype("FTMUSDT"),
      genOptiontype("FTTUSDT"),
      genOptiontype("GTCUSDT"),
      genOptiontype("HBARUSDT"),
      genOptiontype("HFTUSDT"),
      genOptiontype("HIFIUSDT"),
      genOptiontype("HIVEUSDT"),
      genOptiontype("HOTUSDT"),
      genOptiontype("IDUSDT"),
      genOptiontype("ILVUSDT"),
      genOptiontype("IMXUSDT"),
      genOptiontype("INJUSDT"),
      genOptiontype("IOSTUSDT"),
      genOptiontype("IOTXUSDT"),
      genOptiontype("IQUSDT"),
      genOptiontype("JASMYUSDT"),
      genOptiontype("JOEUSDT"),
      genOptiontype("JTOUSDT"),
      genOptiontype("JUPUSDT"),
      genOptiontype("KAVAUSDT"),
      genOptiontype("KMDUSDT"),
      genOptiontype("KNCUSDT"),
      genOptiontype("KP3RUSDT"),
      genOptiontype("KSMUSDT"),
      genOptiontype("LDOUSDT"),
      genOptiontype("LEVERUSDT"),
      genOptiontype("LINAUSDT"),
      genOptiontype("LINKUSDT"),
      genOptiontype("LOOMUSDT"),
      genOptiontype("LPTUSDT"),
      genOptiontype("LQTYUSDT"),
      genOptiontype("LRCUSDT"),
      genOptiontype("LSKUSDT"),
      genOptiontype("LTCUSDT"),
      genOptiontype("LTOUSDT"),
      genOptiontype("LUNCUSDT"),
      genOptiontype("MAGICUSDT"),
      genOptiontype("MANAUSDT"),
      genOptiontype("MANTAUSDT"),
      genOptiontype("MASKUSDT"),
      genOptiontype("MATICUSDT"),
      genOptiontype("MAVUSDT"),
      genOptiontype("MBLUSDT"),
      genOptiontype("MDTUSDT"),
      genOptiontype("MEMEUSDT"),
      genOptiontype("METISUSDT"),
      genOptiontype("MKRUSDT"),
      genOptiontype("MLNUSDT"),
      genOptiontype("MOVRUSDT"),
      genOptiontype("MTLUSDT"),
      genOptiontype("NEARUSDT"),
      genOptiontype("NEOUSDT"),
      genOptiontype("NEXOUSDT"),
      genOptiontype("NFPUSDT"),
      genOptiontype("NKNUSDT"),
      genOptiontype("NMRUSDT"),
      genOptiontype("NTRNUSDT"),
      genOptiontype("NULSUSDT"),
      genOptiontype("OAXUSDT"),
      genOptiontype("OCEANUSDT"),
      genOptiontype("OGNUSDT"),
      genOptiontype("OGUSDT"),
      genOptiontype("OMGUSDT"),
      genOptiontype("OMUSDT"),
      genOptiontype("OMNIUSDT"),
    ],
  },
  {
    label: "Altcoins +300",
    pairs: [
      genOptiontype("ALGOUSDT"),
      genOptiontype("GALAUSDT"),
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
      genOptiontype("XRPUSDT"),
      genOptiontype("UTKUSDT"),
      genOptiontype("VICUSDT"),
      genOptiontype("VGXUSDT"),
      genOptiontype("TROYUSDT"),
      genOptiontype("UNFIUSDT"),
      genOptiontype("THETAUSDT"),
      genOptiontype("TFUELUSDT"),
      genOptiontype("TAOUSDT"),
      genOptiontype("AERGOUSDT"),
      genOptiontype("ACMUSDT"),
      genOptiontype("AGIXUSDT"),
      genOptiontype("AIUSDT"),
      genOptiontype("ALICEUSDT"),
      genOptiontype("ALPACAUSDT"),
      genOptiontype("ALPINEUSDT"),
      genOptiontype("ALTUSDT"),
      genOptiontype("ANKRUSDT"),
      genOptiontype("APEUSDT"),
      genOptiontype("APTUSDT"),
      genOptiontype("BONKUSDT"),
      genOptiontype("CITYUSDT"),
      genOptiontype("CTSIUSDT"),
      genOptiontype("CVXUSDT"),
      genOptiontype("DARUSDT"),
      genOptiontype("DATAUSDT"),
      genOptiontype("DCRUSDT"),
      genOptiontype("DENTUSDT"),
      genOptiontype("DEXEUSDT"),
      genOptiontype("DFUSDT"),
      genOptiontype("DGBUSDT"),
      genOptiontype("DIAUSDT"),
      genOptiontype("EDUUSDT"),
      genOptiontype("ENAUSDT"),
      genOptiontype("EPXUSDT"),
      genOptiontype("ERNUSDT"),
      genOptiontype("ETHFIUSDT"),
      genOptiontype("FIROUSDT"),
      genOptiontype("FLMUSDT"),
      genOptiontype("FLUXUSDT"),
      genOptiontype("FORTHUSDT"),
      genOptiontype("FORUSDT"),
      genOptiontype("FXSUSDT"),
      genOptiontype("GALUSDT"),
      genOptiontype("GFTUSDT"),
      genOptiontype("GHSTUSDT"),
      genOptiontype("GLMRUSDT"),
      genOptiontype("GLMUSDT"),
      genOptiontype("GMTUSDT"),
      genOptiontype("GMXUSDT"),
      genOptiontype("GNOUSDT"),
      genOptiontype("GRTUSDT"),
      genOptiontype("JSTUSDT"),
      genOptiontype("BARUSDT"),
      genOptiontype("BEAMXUSDT"),
      genOptiontype("BELUSDT"),
      genOptiontype("BETAUSDT"),
      genOptiontype("BICOUSDT"),
      genOptiontype("BIFIUSDT"),
      genOptiontype("BLZUSDT"),
      genOptiontype("BLURUSDT"),
      genOptiontype("BNTUSDT"),
      genOptiontype("BNXUSDT"),
      genOptiontype("BOMEUSDT"),
      genOptiontype("BONDUSDT"),
      genOptiontype("BSWUSDT"),
      genOptiontype("BTTCUSDT"),
      genOptiontype("BURGERUSDT"),
      genOptiontype("C98USDT"),
      genOptiontype("CAKEUSDT"),
      genOptiontype("CELOUSDT"),
      genOptiontype("CELRUSDT"),
      genOptiontype("CFXUSDT"),
      genOptiontype("CHESSUSDT"),
      genOptiontype("CHRUSDT"),
      genOptiontype("CHZUSDT"),
      genOptiontype("CKBUSDT"),
      genOptiontype("CLVUSDT"),
      genOptiontype("COMBOUSDT"),
      genOptiontype("COMPUSDT"),
      genOptiontype("COSUSDT"),
      genOptiontype("COTIUSDT"),
      genOptiontype("CREAMUSDT"),
      genOptiontype("CRVUSDT"),
      genOptiontype("CTKUSDT"),
      genOptiontype("CTXCUSDT"),
      genOptiontype("CVPUSDT"),
      genOptiontype("CYBERUSDT"),
      genOptiontype("DASHUSDT"),
      genOptiontype("DODOUSDT"),
      genOptiontype("DOTUSDT"),
      genOptiontype("DYDXUSDT"),
      genOptiontype("DYMUSDT"),
      genOptiontype("ENJUSDT"),
      genOptiontype("EOSUSDT"),
      genOptiontype("FIDAUSDT"),
      genOptiontype("FILUSDT"),
      genOptiontype("FTMUSDT"),
      genOptiontype("FTTUSDT"),
      genOptiontype("GTCUSDT"),
      genOptiontype("HBARUSDT"),
      genOptiontype("HFTUSDT"),
      genOptiontype("HIFIUSDT"),
      genOptiontype("HIVEUSDT"),
      genOptiontype("HOTUSDT"),
      genOptiontype("IDUSDT"),
      genOptiontype("ILVUSDT"),
      genOptiontype("IMXUSDT"),
      genOptiontype("INJUSDT"),
      genOptiontype("IOSTUSDT"),
      genOptiontype("IOTXUSDT"),
      genOptiontype("IQUSDT"),
      genOptiontype("JASMYUSDT"),
      genOptiontype("JOEUSDT"),
      genOptiontype("JTOUSDT"),
      genOptiontype("JUPUSDT"),
      genOptiontype("KAVAUSDT"),
      genOptiontype("KMDUSDT"),
      genOptiontype("KNCUSDT"),
      genOptiontype("KP3RUSDT"),
      genOptiontype("KSMUSDT"),
      genOptiontype("LDOUSDT"),
      genOptiontype("LEVERUSDT"),
      genOptiontype("LINAUSDT"),
      genOptiontype("LINKUSDT"),
      genOptiontype("LOOMUSDT"),
      genOptiontype("LPTUSDT"),
      genOptiontype("LQTYUSDT"),
      genOptiontype("LRCUSDT"),
      genOptiontype("LSKUSDT"),
      genOptiontype("LTCUSDT"),
      genOptiontype("LTOUSDT"),
      genOptiontype("LUNCUSDT"),
      genOptiontype("MAGICUSDT"),
      genOptiontype("MANAUSDT"),
      genOptiontype("MANTAUSDT"),
      genOptiontype("MASKUSDT"),
      genOptiontype("MATICUSDT"),
      genOptiontype("MAVUSDT"),
      genOptiontype("MBLUSDT"),
      genOptiontype("MDTUSDT"),
      genOptiontype("MEMEUSDT"),
      genOptiontype("METISUSDT"),
      genOptiontype("MKRUSDT"),
      genOptiontype("MLNUSDT"),
      genOptiontype("MOVRUSDT"),
      genOptiontype("MTLUSDT"),
      genOptiontype("NEARUSDT"),
      genOptiontype("NEOUSDT"),
      genOptiontype("NEXOUSDT"),
      genOptiontype("NFPUSDT"),
      genOptiontype("NKNUSDT"),
      genOptiontype("NMRUSDT"),
      genOptiontype("NTRNUSDT"),
      genOptiontype("NULSUSDT"),
      genOptiontype("OAXUSDT"),
      genOptiontype("OCEANUSDT"),
      genOptiontype("OGNUSDT"),
      genOptiontype("OGUSDT"),
      genOptiontype("OMGUSDT"),
      genOptiontype("OMUSDT"),
      genOptiontype("OMNIUSDT"),
      genOptiontype("OMUSDT"),
      genOptiontype("ONEUSDT"),
      genOptiontype("ONGUSDT"),
      genOptiontype("ONTUSDT"),
      genOptiontype("OOKIUSDT"),
      genOptiontype("OPUSDT"),
      genOptiontype("ORDIUSDT"),
      genOptiontype("ORNUSDT"),
      genOptiontype("OSMOUSDT"),
      genOptiontype("OXTUSDT"),
      genOptiontype("PAXGUSDT"),
      genOptiontype("PDAUSDT"),
      genOptiontype("PENDLEUSDT"),
      genOptiontype("PEOPLEUSDT"),
      genOptiontype("PEPEUSDT"),
      genOptiontype("PERPUSDT"),
      genOptiontype("PHAUSDT"),
      genOptiontype("PHBUSDT"),
      genOptiontype("PIVXUSDT"),
      genOptiontype("PIXELUSDT"),
      genOptiontype("POLSUSDT"),
      genOptiontype("POLYXUSDT"),
      genOptiontype("PONDUSDT"),
      genOptiontype("PORTALUSDT"),
      genOptiontype("PORTOUSDT"),
      genOptiontype("POWRUSDT"),
      genOptiontype("PROMUSDT"),
      genOptiontype("PROSUSDT"),
      genOptiontype("PSGUSDT"),
      genOptiontype("PUNDIXUSDT"),
      genOptiontype("PYRUSDT"),
      genOptiontype("PYTHUSDT"),
      genOptiontype("QIUSDT"),
      genOptiontype("QKCUSDT"),
      genOptiontype("QNTUSDT"),
      genOptiontype("QTUMUSDT"),
      genOptiontype("QUICKUSDT"),
      genOptiontype("RADUSDT"),
      genOptiontype("RAREUSDT"),
      genOptiontype("RAYUSDT"),
      genOptiontype("RDNTUSDT"),
      genOptiontype("REEFUSDT"),
      genOptiontype("REIUSDT"),
      genOptiontype("RENUSDT"),
      genOptiontype("REQUSDT"),
      genOptiontype("REZUSDT"),
      genOptiontype("RIFUSDT"),
      genOptiontype("RLCUSDT"),
      genOptiontype("RNDRUSDT"),
    ],
  },
];
