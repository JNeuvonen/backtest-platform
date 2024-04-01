package main

import (
	"testing"
)

func TestMarketOrder(t *testing.T) {
	tradingConfig := getTradingConfig()

	binanceClient := NewBinanceClient(tradingConfig)
	binanceClient.SendOrder("BTCUSDT", "BUY", "MARKET", 0.001, true)
}
