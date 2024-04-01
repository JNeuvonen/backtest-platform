package main

import (
	"context"
	"fmt"
	"testing"

	binance_connector "github.com/binance/binance-connector-go"
)

func TestMarketOrder(t *testing.T) {
	apiKey, apiSecret, baseURL := getBinanceKeys()
	client := binance_connector.NewClient(apiKey, apiSecret, baseURL)

	assertIsTestnet(client.BaseURL)

	newOrder, err := client.NewCreateOrderService().Symbol("BTCUSDT").
		Side("BUY").Type("MARKET").Quantity(0.001).
		Do(context.Background())
	if err != nil {
		t.Fatalf("Failed to create order: %v", err)
	}
	fmt.Println(binance_connector.PrettyPrint(newOrder))
}
