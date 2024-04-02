package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMarketOrder(t *testing.T) {
	tradingConfig := getTradingConfig()

	binanceClient := NewBinanceClient(tradingConfig)
	binanceClient.SendOrder("BTCUSDT", "BUY", "MARKET", 0.001, true)
}

func TestCallingPredServer(t *testing.T) {
	predServConfig := getPredServerConfig()
	headers := map[string]string{
		"X-API-KEY": predServConfig.API_KEY,
	}
	client := NewHttpClient(predServConfig.URI, headers)
	_, err := client.Get("")
	assert.Nil(t, err, "Error calling prediction server: %v", err)
}
