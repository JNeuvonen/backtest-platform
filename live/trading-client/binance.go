package main

import (
	"context"

	"github.com/adshao/go-binance/v2"
)

type BinanceClient struct {
	Client *binance.Client
}

func NewBinanceClient() *BinanceClient {
	apiKey, apiSecret := getBinanceKeys()
	client := getBinanceClient(apiKey, apiSecret)

	return &BinanceClient{
		Client: client,
	}
}

func (this *BinanceClient) GetBookDepth(symbol string) (*binance.DepthResponse, error) {
	res, err := this.Client.NewDepthService().Symbol(symbol).
		Do(context.Background())
	return res, err
}
