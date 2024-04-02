package main

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMarketOrder(t *testing.T) {
	tradingConfig := GetTradingConfig()

	binanceClient := NewBinanceClient(tradingConfig)
	binanceClient.SendOrder("BTCUSDT", "BUY", "MARKET", 0.001, true)
}

func TestCallingPredServer(t *testing.T) {
	predServConfig := GetPredServerConfig()
	headers := map[string]string{
		"X-API-KEY": predServConfig.API_KEY,
	}
	client := NewHttpClient(predServConfig.URI, headers)
	_, err := client.Get("")
	assert.Nil(t, err, "Error calling prediction server: %v", err)
}

func TestFetchingStrategies(t *testing.T) {
	strategies, err := FetchStrategies()
	assert.Nil(t, err, "Error fetching strategies: %v", err)

	for _, strategy := range strategies {
		fmt.Printf(
			"Strategy ID: %d, Symbol: %s, Priority: %d\n",
			strategy.ID,
			strategy.Symbol,
			strategy.Priority,
		)
	}
}
