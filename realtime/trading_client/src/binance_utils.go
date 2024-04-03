package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"

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
		CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), "exception")
		return
	}

	fmt.Println(binance_connector.PrettyPrint(order))
}

func (bc *BinanceClient) FetchBalances() *binance_connector.AccountResponse {
	account, err := bc.client.NewGetAccountService().Do(context.Background())
	if err != nil {
		CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), "exception")
		return nil
	}
	return account
}

func (bc *BinanceClient) GetCurrentAccountWorthInUSDT() (float64, error) {
	account, err := bc.client.NewGetAccountService().Do(context.Background())
	if err != nil {
		CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), "exception")
		return 0, err
	}

	totalWorth := 0.0
	for _, balance := range account.Balances {
		fmt.Println(balance.Free)
		if balance.Asset == "USDT" {
			worth, _ := strconv.ParseFloat(balance.Free, 64)
			totalWorth += worth
		} else {
			symbol := balance.Asset + "USDT"
			priceResponse, err := bc.client.NewAvgPriceService().Symbol(symbol).Do(context.Background())
			if priceResponse == nil || err != nil {
				continue
			}

			assetWorth, _ := strconv.ParseFloat(balance.Free, 64)
			price, _ := strconv.ParseFloat(priceResponse.Price, 64)
			totalWorth += assetWorth * price
		}
	}
	return totalWorth, nil
}

func (bc *BinanceClient) FetchLatestPrice(symbol string) (float64, error) {
	priceRes, err := bc.client.NewTickerPriceService().Symbol(symbol).Do(context.Background())
	if err != nil {
		CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), "exception")
		return 0, err
	}

	price, err := strconv.ParseFloat(priceRes.Price, 64)
	if err != nil {
		return 0, err
	}

	return price, nil
}

func GetAllUSDTPrices() ([]SymbolInfoSimple, error) {
	baseURL := GetBinanceAPIBaseURL()
	resp, err := http.Get(baseURL + V3_PRICE)
	if err != nil {
		CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), "exception")
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), "exception")
		return nil, err
	}

	var prices []SymbolInfoSimple
	err = json.Unmarshal(body, &prices)
	if err != nil {
		CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), "exception")
		return nil, err
	}

	var usdtPairs []SymbolInfoSimple
	for _, price := range prices {
		if len(price.Symbol) > 4 && price.Symbol[len(price.Symbol)-4:] == "USDT" {
			usdtPairs = append(usdtPairs, price)
		}
	}

	return usdtPairs, nil
}
