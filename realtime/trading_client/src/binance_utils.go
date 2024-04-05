package main

import (
	"context"
	"encoding/json"
	"errors"
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

func (bc *BinanceClient) SendSpotOrder(
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

func getAllUSDTPrices() ([]SymbolInfoSimple, error) {
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

func (bc *BinanceClient) GetPortfolioValueInUSDT() (float64, error) {
	prices, pricesErr := getAllUSDTPrices()

	if pricesErr != nil {
		CreateCloudLog(NewFmtError(pricesErr, CaptureStack()).Error(), "exception")
		return 0.0, pricesErr
	}
	balances := bc.FetchBalances()

	if balances == nil {
		CreateCloudLog(NewFmtError(pricesErr, CaptureStack()).Error(), "exception")
		return 0.0, errors.New(
			"Could not fetch balances: bc.FetchBalances() in GetPortfolioValueInUSDT()",
		)

	}
	accountValueUSDT := 0.0

	for _, balance := range balances.Balances {
		free := ParseToFloat64(balance.Free, 0.0)
		if free >= 0.0 {
			symbolInfo := FindListItem[SymbolInfoSimple](
				prices,
				func(i SymbolInfoSimple) bool { return i.Symbol == balance.Asset+"USDT" },
			)

			if symbolInfo == nil {
				continue
			}

			accountValueUSDT += ParseToFloat64(symbolInfo.Price, 0.0) * free
		}
	}

	return accountValueUSDT, nil
}

func GetFreeBalanceForAsset(balances []binance_connector.Balance, asset string) (string, error) {
	for _, balance := range balances {
		if balance.Asset == asset {
			return balance.Free, nil
		}
	}
	return "", fmt.Errorf("asset not found")
}

func (bc *BinanceClient) GetAssetDebtRatioUSDT() float64 {
	crossMarginDetailsRes, err := bc.client.NewCrossMarginAccountDetailService().
		Do(context.Background())
	if err != nil {
		CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), "exception")
		return 0.0
	}

	usdtPrices, err := getAllUSDTPrices()
	if err != nil {
		CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), "exception")
		return 0.0
	}

	totalAssetsUSDT, totalDebtUSDT := 0.0, 0.0

	for _, userAsset := range crossMarginDetailsRes.UserAssets {
		netAsset := ParseToFloat64(userAsset.NetAsset, 0.0)
		borrowed := ParseToFloat64(userAsset.Borrowed, 0.0)
		interest := ParseToFloat64(userAsset.Interest, 0.0)
		totalDebt := borrowed + interest

		assetUSDTValue, debtUSDTValue := 0.0, 0.0

		if userAsset.Asset == "USDT" {
			assetUSDTValue = netAsset
			debtUSDTValue = totalDebt
		} else {
			symbolInfo := FindListItem[SymbolInfoSimple](
				usdtPrices,
				func(i SymbolInfoSimple) bool { return i.Symbol == userAsset.Asset+"USDT" },
			)

			if symbolInfo != nil {
				usdtPrice := ParseToFloat64(symbolInfo.Price, 0.0)
				assetUSDTValue = netAsset * usdtPrice
				debtUSDTValue = totalDebt * usdtPrice
			}
		}

		totalAssetsUSDT += assetUSDTValue
		totalDebtUSDT += debtUSDTValue
	}

	if totalDebtUSDT == 0 {
		return 0.0
	}

	return totalDebtUSDT / totalAssetsUSDT
}

func (bc *BinanceClient) GetAccountNetValueUSDT() (float64, error) {
	crossMarginDetailsRes, err := bc.client.NewCrossMarginAccountDetailService().
		Do(context.Background())
	if err != nil {
		CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), "exception")
		fmt.Println(err)
		return 0.0, err
	}

	usdtPrices, err := getAllUSDTPrices()
	if err != nil {
		CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), "exception")
		fmt.Println(err)
		return 0.0, err
	}

	accountNetUSDTValue := 0.0

	for _, userAsset := range crossMarginDetailsRes.UserAssets {
		netAsset := ParseToFloat64(userAsset.NetAsset, 0.0)

		if netAsset > 0.0 {

			if userAsset.Asset == "USDT" {
				accountNetUSDTValue += netAsset
				continue
			}

			symbolInfo := FindListItem[SymbolInfoSimple](
				usdtPrices,
				func(i SymbolInfoSimple) bool { return i.Symbol == userAsset.Asset+"USDT" },
			)

			if symbolInfo == nil {
				continue
			}

			accountNetUSDTValue += netAsset * ParseToFloat64(symbolInfo.Price, 0.0)
		}
	}

	return accountNetUSDTValue, nil
}

func (bc *BinanceClient) GetAccountDebtInUSDT() (float64, error) {
	crossMarginDetailsRes, err := bc.client.NewCrossMarginAccountDetailService().
		Do(context.Background())
	if err != nil {
		CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), "exception")
		fmt.Println(err)
		return 0.0, err
	}

	usdtPrices, err := getAllUSDTPrices()
	if err != nil {
		CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), "exception")
		fmt.Println(err)
		return 0.0, err
	}

	totalDebtUSDT := 0.0

	for _, userAsset := range crossMarginDetailsRes.UserAssets {
		if ParseToFloat64(userAsset.Borrowed, 0.0) > 0.0 {

			if userAsset.Asset == "USDT" {
				totalDebtUSDT += ParseToFloat64(userAsset.Borrowed, 0.0)
				continue
			}

			symbolInfo := FindListItem[SymbolInfoSimple](
				usdtPrices,
				func(i SymbolInfoSimple) bool { return i.Symbol == userAsset.Asset+"USDT" },
			)

			if symbolInfo == nil {
				continue
			}

			totalDebtUSDT += ParseToFloat64(
				userAsset.Borrowed,
				0.0,
			) * ParseToFloat64(
				symbolInfo.Price,
				0.0,
			)
		}
	}

	return totalDebtUSDT, nil
}

func (bc *BinanceClient) GetAccountDebtRatio() (float64, error) {
	totalAccountDebtUSDT, err := bc.GetAccountDebtInUSDT()
	if err != nil {

		CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), "exception")
		fmt.Println(err)
		return 0.0, err
	}

	totalAccountValueUSDT, err := bc.GetPortfolioValueInUSDT()
	if err != nil {
		CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), "exception")
		fmt.Println(err)
		return 0.0, err
	}

	if totalAccountValueUSDT == 0 {
		err := errors.New("totalAccountValueUSDT was 0, GetAccountDebtRatio()")
		CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), "exception")
		return 0.0, err
	}
	return totalAccountDebtUSDT / totalAccountValueUSDT, nil
}

func (bc *BinanceClient) TakeMarginLoan(asset string, quantity float64) error {
	if !GetAllowedToSendOrders() {
		ret := errors.New("Environment is not allowed to make borrows")
		CreateCloudLog(
			NewFmtError(
				ret,
				CaptureStack(),
			).Error(),
			"info",
		)
		return ret
	}

	fmt.Println(asset, quantity)
	_, err := bc.client.NewBorrowService().
		Asset(asset).
		Amount(quantity).
		Do(context.Background())
	if err != nil {
		CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), "exception")
		return err
	}

	return nil
}

func (bc *BinanceClient) NewMarginOrder(
	symbol string,
	quantity float64,
	side string,
	orderType string,
	strat Strategy,
) error {
	if !GetAllowedToSendOrders() {
		ret := errors.New("Environment is not allowed to send orders")
		CreateCloudLog(
			NewFmtError(
				ret,
				CaptureStack(),
			).Error(),
			"info",
		)
		return ret
	}

	res, err := bc.client.NewMarginAccountNewOrderService().
		Symbol(symbol).
		Side(side).
		OrderType(orderType).
		NewOrderRespType("FULL").
		Quantity(quantity).Do(context.Background())
	if err != nil {
		CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), "exception")
		return err
	}

	fullRes, ok := res.(*binance_connector.MarginAccountNewOrderResponseFULL)
	if !ok {
		err = errors.New("Type assertion for response to MarginAccountNewOrderResponseFULL failed")
		CreateCloudLog(
			NewFmtError(
				err,
				CaptureStack(),
			).Error(), "exception")
		return err
	}

	tradeID := CreateTradeEntry(map[string]interface{}{
		"open_price":                fullRes.Price,
		"open_time_ms":              fullRes.TransactTime,
		"quantity":                  fullRes.ExecutedQty,
		"cumulative_quote_quantity": fullRes.CumulativeQuoteQty,
		"direction":                 "SHORT",
		"strategy_id":               int32(strat.ID),
	})

	UpdateStrategy(int32(strat.ID), map[string]interface{}{
		"price_on_trade_open":        fullRes.Price,
		"time_on_trade_open_ms":      fullRes.TransactTime,
		"klines_left_till_autoclose": strat.MaximumKlinesHoldTime,
		"active_trade_id":            tradeID,
	})

	return nil
}
