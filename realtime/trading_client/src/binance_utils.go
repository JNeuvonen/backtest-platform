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
			tradingConfig.ApiKey,
			tradingConfig.ApiSecret,
			tradingConfig.BaseUrl,
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
	order, err := bc.client.NewCreateOrderService().Symbol(symbol).
		Side(side).Type(orderType).Quantity(quantity).
		Do(context.Background())
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println(binance_connector.PrettyPrint(order))
}

func (bc *BinanceClient) FetchBalances() {
	account, err := bc.client.NewGetAccountService().Do(context.Background())
	if err != nil {
		fmt.Println(err)
		return
	}

	for _, balance := range account.Balances {
		fmt.Printf("Asset: %s, Free: %s, Locked: %s\n", balance.Asset, balance.Free, balance.Locked)
	}
}
