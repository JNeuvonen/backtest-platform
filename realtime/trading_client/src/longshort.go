package main

import (
	"errors"
	"fmt"
	"math"
	"strconv"

	binance_connector "live/trading-client/src/thirdparty/binance-connector-go"
)

func getLongShortEnterUSDTSize(
	bc *BinanceClient,
	account *Account,
	maxAllocationRatio float64,
) float64 {
	accUSDTValue, _ := bc.GetAccountNetValueUSDT()

	if accUSDTValue == 0.0 {
		return 0.0
	}

	accDebtRatio := bc.GetAssetDebtRatioUSDT()

	if accDebtRatio+maxAllocationRatio > account.MaxDebtRatio {
		CreateCloudLog(
			NewFmtError(
				errors.New(
					"Acc debt ratio would surpass account maximum, aborting long/short enter",
				),
				CaptureStack(),
			).Error(),
			LOG_INFO,
		)
		return 0.0
	}

	allocatedRatio := math.Min(account.MaxDebtRatio-maxAllocationRatio, maxAllocationRatio)
	return allocatedRatio * accUSDTValue
}

func initPairTradeShort(
	bc *BinanceClient,
	usdtEnterSize float64,
	pair *LongShortPair,
) (*binance_connector.MarginAccountNewOrderResponseFULL, error) {
	price, err := bc.FetchLatestPrice(pair.SellSymbol)
	if err != nil {
		CreateCloudLog(
			NewFmtError(
				err,
				CaptureStack(),
			).Error(),
			LOG_EXCEPTION,
		)
		return nil, err
	}

	quantity := GetBaseQuantity(usdtEnterSize, price, int32(pair.SellQtyPrecision))

	err = bc.TakeMarginLoan(pair.SellBaseAsset, quantity)

	if err == nil {
		res := bc.NewMarginOrder(pair.SellSymbol, quantity, ORDER_SELL, MARKET_ORDER)
		if res != nil {
			return res, nil
		}
		UpdatePairTradeEnterError(pair.ID)
		return nil, errors.New("Failed to submit sell order for long/short strategy")
	}

	UpdatePairTradeEnterError(pair.ID)
	return nil, errors.New("Failed to take loan for long/short strategy")
}

func initPairTradeLong(
	bc *BinanceClient,
	usdtEnterSize float64,
	pair *LongShortPair,
) (*binance_connector.MarginAccountNewOrderResponseFULL, error) {
	price, _ := bc.FetchLatestPrice(pair.BuySymbol)

	quantity := GetBaseQuantity(usdtEnterSize, price, int32(pair.SellQtyPrecision))

	res := bc.NewMarginOrder(pair.BuySymbol, quantity, ORDER_BUY, MARKET_ORDER)

	if res != nil {
		return res, nil
	}

	UpdatePairTradeEnterError(pair.ID)
	return nil, errors.New("Failed to enter to long position for long/short strategy")
}

func updatePredServOnLongShortEnter(
	predServClient *HttpClient,
	pair *LongShortPair,
	sellSideRes *binance_connector.MarginAccountNewOrderResponseFULL,
	buySideRes *binance_connector.MarginAccountNewOrderResponseFULL,
) {
}

func enterLongShortTrade(
	bc *BinanceClient,
	predServClient *HttpClient,
	group *LongShortGroup,
	pair *LongShortPair,
) {
	accountName := GetAccountName()
	account, _ := predServClient.FetchAccount(accountName)
	maxAllocationRatio := group.MaxLeverageRatio / float64(group.MaxSimultaneousPositions)
	usdtEnterSize := getLongShortEnterUSDTSize(bc, &account, maxAllocationRatio)

	sellSideRes, err := initPairTradeShort(bc, usdtEnterSize, pair)

	if err == nil {
		sellOrderQuoteQty, _ := strconv.ParseFloat(sellSideRes.CumulativeQuoteQty, 64)
		buySideRes, err := initPairTradeLong(bc, sellOrderQuoteQty, pair)

		if err == nil {
			updatePredServOnLongShortEnter(predServClient, pair, sellSideRes, buySideRes)
		}
	}
}

func ProcessLongShortGroup(bc *BinanceClient, predServClient *HttpClient, group *LongShortGroup) {
	pairs := predServClient.FetchLongShortPairs(group.ID)

	for _, pair := range pairs {

		if pair.InPosition {
		}

		if !pair.InPosition && !pair.IsTradeFinished && !pair.ErrorInEntering {
			enterLongShortTrade(bc, predServClient, group, &pair)
		}
	}
}
