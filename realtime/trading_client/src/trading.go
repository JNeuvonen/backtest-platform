package main

import (
	"errors"
	"fmt"
	"math"

	binance_connector "github.com/binance/binance-connector-go"
)

func shouldStopLossClose(strat Strategy, price float64) bool {
	if strat.IsShortSellingStrategy {
		threshold := 1 + (strat.StopLossThresholdPerc / 100)
		return price > strat.PriceOnTradeOpen*threshold
	} else {
		threshold := 1 - (strat.StopLossThresholdPerc / 100)
		return price < strat.PriceOnTradeOpen*threshold
	}
}

func shouldTimebasedClose(strat Strategy) bool {
	if strat.UseTimeBasedClose {
		return strat.KlinesLeftTillAutoclose == 0
	}
	return false
}

func shouldProfitBasedClose(strat Strategy, price float64) bool {
	profitThreshold := 1 - (strat.TakeProfitThresholdPerc / 100)

	if strat.IsShortSellingStrategy {
		return price < strat.PriceOnTradeOpen*profitThreshold
	} else {
		return price > strat.PriceOnTradeOpen*(1+strat.TakeProfitThresholdPerc/100)
	}
}

func ShouldCloseTrade(bc *BinanceClient, strat Strategy) bool {
	price, err := bc.FetchLatestPrice(strat.Symbol)
	if err != nil {
		return false
	}

	return shouldStopLossClose(strat, price) || shouldTimebasedClose(strat) ||
		shouldProfitBasedClose(strat, price) || strat.ShouldCloseTrade
}

func ShouldEnterTrade(strat Strategy) bool {
	currTimeMs := GetTimeInMs()

	if strat.TimeOnTradeOpenMs-int64(currTimeMs) >= int64(strat.MinimumTimeBetweenTradesMs) {
		return strat.ShouldEnterTrade
	}
	return false
}

func CloseStrategyTrade(bc *BinanceClient, strat Strategy) {
}

func calculateShortStratBetSizeUSDT(
	bc *BinanceClient,
	account Account,
	strat Strategy,
	accUSDTValue float64,
) float64 {
	accDebtRatio := bc.GetAssetDebtRatioUSDT()

	if accDebtRatio > account.MaxDebtRatio {
		CreateCloudLog(
			NewFmtError(errors.New("accDebtRatio > account.MaxDebtRatio"), CaptureStack()).Error(),
			"info",
		)
		return 0.0
	}

	maxAllocatedUSDTValue := (strat.AllocatedSizePerc / 100) * accUSDTValue
	debtInUSDT, _ := bc.GetAccountDebtInUSDT()

	if maxAllocatedUSDTValue+debtInUSDT/accUSDTValue < account.MaxDebtRatio {
		return maxAllocatedUSDTValue
	} else {
		return math.Min((account.MaxDebtRatio-accDebtRatio)*accUSDTValue, maxAllocatedUSDTValue)
	}
}

func calculateLongStratBetSizeUSDT(
	strat Strategy,
	balances []binance_connector.Balance,
	accUSDTValue float64,
) float64 {
	freeUSDT, err := GetFreeBalanceForAsset(balances, "USDT")
	if err != nil {
		CreateCloudLog(
			NewFmtError(err, CaptureStack()).Error(),
			"exception",
		)
		return 0.0
	}

	maxAllocatedUSDTValue := (strat.AllocatedSizePerc / 100) * accUSDTValue
	return math.Min(ParseToFloat64(freeUSDT, 0.0), maxAllocatedUSDTValue)
}

func GetStrategyAvailableBetsizeUSDT(bc *BinanceClient, strat Strategy, account Account) float64 {
	accUSDTValue, err := bc.GetAccountNetValueUSDT()
	if err != nil {
		CreateCloudLog(
			NewFmtError(err, CaptureStack()).Error(),
			"exception",
		)
		return 0.0
	}

	if accUSDTValue == 0.0 {
		CreateCloudLog(
			NewFmtError(err, CaptureStack()).Error(),
			"exception",
		)
		return 0.0
	}

	balances := bc.FetchBalances()

	if balances != nil {
		if strat.IsShortSellingStrategy {
			return calculateShortStratBetSizeUSDT(bc, account, strat, accUSDTValue)
		} else {
			return calculateLongStratBetSizeUSDT(strat, balances.Balances, accUSDTValue)
		}
	}

	CreateCloudLog(
		NewFmtError(
			errors.New(
				"GetStrategyAvailableBetsizeUSDT(): balances := bc.FetchBalances() was unexpectedly nil",
			),
			CaptureStack(),
		).Error(),
		"exception",
	)
	return 0.0
}

func OpenLongTrade(strat Strategy, bc *BinanceClient, sizeUSDT float64) {
	price, err := bc.FetchLatestPrice(strat.Symbol)
	if err != nil {
		CreateCloudLog(
			NewFmtError(
				err,
				CaptureStack(),
			).Error(),
			"exception",
		)
		return
	}

	quantity := GetBaseQuantity(sizeUSDT, price, int32(strat.TradeQuantityPrecision))
	fmt.Println(quantity)
}

func OpenShortTrade(strat Strategy, bc *BinanceClient, sizeUSDT float64) {
	price, err := bc.FetchLatestPrice(strat.Symbol)
	if err != nil {
		CreateCloudLog(
			NewFmtError(
				err,
				CaptureStack(),
			).Error(),
			"exception",
		)
		return
	}

	quantity := GetBaseQuantity(sizeUSDT, price, int32(strat.TradeQuantityPrecision))

	err = bc.TakeMarginLoan(strat.BaseAsset, quantity)

	fmt.Println(err)

	if err == nil {
		bc.NewMarginOrder(strat.Symbol, quantity, "SELL", "MARKET", strat)
	}
}

func EnterStrategyTrade(bc *BinanceClient, strat Strategy, account Account) {
	sizeUSDT := GetStrategyAvailableBetsizeUSDT(bc, strat, account)

	if strat.IsShortSellingStrategy {
		OpenShortTrade(strat, bc, sizeUSDT)
	} else {
		OpenLongTrade(strat, bc, sizeUSDT)
	}
}

func TradingLoop() {
	predServConfig := GetPredServerConfig()
	tradingConfig := GetTradingConfig()
	accountName := GetAccountName()

	headers := map[string]string{
		"X-API-KEY": predServConfig.API_KEY,
	}
	predServClient := NewHttpClient(predServConfig.URI, headers)
	binanceClient := NewBinanceClient(tradingConfig)

	for {
		balances := binanceClient.FetchBalances()

		for _, balance := range balances.Balances {
			fmt.Println(balance.Free)
		}

		strategies := predServClient.FetchStrategies()
		account, _ := predServClient.FetchAccount(accountName)

		for _, strat := range strategies {
			if strat.IsInPosition {
				if ShouldCloseTrade(binanceClient, strat) {
					CloseStrategyTrade(binanceClient, strat)
				}
			} else if !strat.IsInPosition {
				if ShouldEnterTrade(strat) {
					EnterStrategyTrade(binanceClient, strat, account)
				}
			}
		}

		predServClient.CreateCloudLog("Trading loop completed", "info")
	}
}
