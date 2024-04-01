package main

import (
	"context"
	"fmt"

	binance_connector "github.com/binance/binance-connector-go"
)

type BinanceClient struct {
	client        *binance_connector.Client
	tradingConfig TradingConfig
}

func NewBinanceClient(tradingConfig TradingConfig) *BinanceClient {
	return &BinanceClient{
		client: binance_connector.NewClient(
			tradingConfig.TestnetApiKey,
			tradingConfig.TestnetApiSecret,
		),
		tradingConfig: tradingConfig,
	}
}

func (bc *BinanceClient) SendOrder(
	symbol string,
	side string,
	orderType string,
	quantity float64,
	useTestnet bool,
) {
	isUsingMainnet := false

	if !useTestnet && bc.tradingConfig.LiveTradingEnabled == 1 {
		bc.client.APIKey = bc.tradingConfig.ProdApiKey
		bc.client.SecretKey = bc.tradingConfig.ProdApiSecret
		bc.client.BaseURL = MAINNET
		isUsingMainnet = true
	} else {
		bc.client.APIKey = bc.tradingConfig.TestnetApiKey
		bc.client.SecretKey = bc.tradingConfig.TestnetApiSecret
		bc.client.BaseURL = TESTNET
	}

	order, err := bc.client.NewCreateOrderService().Symbol(symbol).
		Side(side).Type(orderType).Quantity(quantity).
		Do(context.Background())
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println(isUsingMainnet)
	fmt.Println(binance_connector.PrettyPrint(order))
}
