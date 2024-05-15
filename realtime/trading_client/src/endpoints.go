package main

import (
	"strconv"
	"strings"
)

const (
	TESTNET = "https://testnet.binance.vision"
	MAINNET = "https://api.binance.com"
)

const (
	PRED_SERV_V1_STRAT             = "v1/strategy"
	PRED_SERV_V1_LONGSHORT         = "v1/longshort"
	PRED_SERV_V1_LONGSHORT_TICKERS = "v1/longshort/tickers/{id}"
	PRED_SERV_V1_LONGSHORT_PAIRS   = "v1/longshort/pairs/{id}"
	PRED_SERV_V1_LONGSHORT_PAIR    = "v1/longshort/pair/{id}"
	PRED_SERV_V1_TRADE_ON_CLOSE    = "v1/strategy/{id}/close-trade"
	PRED_SERV_V1_LOG               = "v1/log"
	PRED_SERV_V1_ACC               = "v1/acc"
	PRED_SERV_V1_TRADE             = "v1/trade"
)

const (
	V3_PRICE = "/api/v3/ticker/price"
)

func GetCloseTradeEndpoint(id int) string {
	return strings.Replace(PRED_SERV_V1_TRADE_ON_CLOSE, "{id}", strconv.Itoa(id), 1)
}

func GetLongShortTickersEndpoint(id int) string {
	return strings.Replace(PRED_SERV_V1_LONGSHORT_TICKERS, "{id}", strconv.Itoa(id), 1)
}

func GetLongShortPairsEndpoint(id int) string {
	return strings.Replace(PRED_SERV_V1_LONGSHORT_PAIRS, "{id}", strconv.Itoa(id), 1)
}

func GetLongShortPairEndpoint(id int) string {
	return strings.Replace(PRED_SERV_V1_LONGSHORT_PAIR, "{id}", strconv.Itoa(id), 1)
}
