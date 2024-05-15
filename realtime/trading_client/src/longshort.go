package main

import (
	"encoding/json"
	"errors"
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
) (*binance_connector.MarginAccountNewOrderResponseFULL, error, float64) {
	price, err := bc.FetchLatestPrice(pair.SellSymbol)
	if err != nil {
		CreateCloudLog(
			NewFmtError(
				err,
				CaptureStack(),
			).Error(),
			LOG_EXCEPTION,
		)
		return nil, err, 0.0
	}

	quantity := GetBaseQuantity(usdtEnterSize, price, int32(pair.SellQtyPrecision))

	err = bc.TakeMarginLoan(pair.SellBaseAsset, quantity)

	if err == nil {
		res := bc.NewMarginOrder(pair.SellSymbol, quantity, ORDER_SELL, MARKET_ORDER)
		if res != nil {
			return res, nil, quantity
		}
		UpdatePairTradeEnterError(pair.ID)
		return nil, errors.New("Failed to submit sell order for long/short strategy"), 0.0
	}

	UpdatePair(pair.ID, map[string]interface{}{
		"is_no_loan_available_err": true,
	})
	return nil, errors.New("Failed to take loan for long/short strategy"), 0.0
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

func closeLong(
	bc *BinanceClient,
	pair *LongShortPair,
) *binance_connector.MarginAccountNewOrderResponseFULL {
	marginBalancesRes := bc.FetchMarginBalances()

	freeBaseAsset := GetFreeBalanceForMarginAsset(marginBalancesRes, pair.BuyBaseAsset)
	closeQuantity := RoundToPrecision(
		math.Min(pair.BuyOpenQtyInBase, freeBaseAsset),
		int32(pair.BuyQtyPrecision),
	)
	res := bc.NewMarginOrder(
		pair.BuySymbol,
		closeQuantity,
		ORDER_SELL,
		MARKET_ORDER,
	)

	return res
}

func getQuoteLoanToClosePairTrade(
	bc *BinanceClient,
	pair *LongShortPair,
	price float64,
	currMaxCloseQuantity float64,
	stratLiabilities float64,
) *binance_connector.MarginAccountNewOrderResponseFULL {
	quoteLoanSize := RoundToPrecision(
		(stratLiabilities-currMaxCloseQuantity)*price,
		0,
	) + USDT_QUOTE_BUFFER

	err := bc.TakeMarginLoan(pair.SellQuoteAsset, quoteLoanSize)

	if err == nil {
		res := bc.NewMarginOrder(
			pair.SellSymbol,
			RoundToPrecision(stratLiabilities, int32(pair.SellQtyPrecision)),
			ORDER_BUY,
			MARKET_ORDER,
		)

		if res != nil {
			bc.RepayMarginLoan(pair.SellBaseAsset, ParseToFloat64(res.ExecutedQty, 0.0))
		}

		return res
	}

	return nil
}

func closeShort(
	bc *BinanceClient,
	pair *LongShortPair,
) *binance_connector.MarginAccountNewOrderResponseFULL {
	marginBalancesRes := bc.FetchMarginBalances()

	stratLiabilities := (pair.DebtOpenQtyInBase + GetInterestInAsset(
		marginBalancesRes,
		pair.SellBaseAsset,
	)) * CLOSE_SHORT_FEES_COEFF
	freeUSDT := GetFreeBalanceForMarginAsset(marginBalancesRes, pair.SellQuoteAsset)

	price, err := bc.FetchLatestPrice(pair.SellSymbol)
	if err != nil {
		CreateCloudLog(
			NewFmtError(
				err,
				CaptureStack(),
			).Error(),
			LOG_EXCEPTION,
		)
		return nil
	}

	currMaxCloseQuantity := SafeDivide(freeUSDT, price)

	if currMaxCloseQuantity >= stratLiabilities {
		res := bc.NewMarginOrder(
			pair.SellSymbol,
			RoundToPrecision(stratLiabilities, int32(pair.SellQtyPrecision)),
			ORDER_BUY,
			MARKET_ORDER,
		)
		if res != nil {
			bc.RepayMarginLoan(pair.SellBaseAsset, ParseToFloat64(res.ExecutedQty, 0.0))
		}
	}

	return getQuoteLoanToClosePairTrade(bc, pair, price, currMaxCloseQuantity, stratLiabilities)
}

func updatePredServOnLongShortEnter(
	predServClient *HttpClient,
	pair *LongShortPair,
	sellSideRes *binance_connector.MarginAccountNewOrderResponseFULL,
	buySideRes *binance_connector.MarginAccountNewOrderResponseFULL,
	debtQuantity float64,
) {
	reqBody := map[string]interface{}{
		"buy_open_qty_in_base":   ParseToFloat64(buySideRes.ExecutedQty, 0.0),
		"buy_open_price":         ParseToFloat64(buySideRes.Price, 0.0),
		"sell_open_price":        ParseToFloat64(sellSideRes.Price, 0.0),
		"sell_open_qty_in_quote": ParseToFloat64(sellSideRes.CumulativeQuoteQty, 0.0),
		"debt_open_qty_in_base":  debtQuantity,
		"buy_open_time_ms":       buySideRes.TransactTime,
		"sell_open_time_ms":      sellSideRes.TransactTime,
		"sell_symbol":            pair.SellSymbol,
		"buy_symbol":             pair.BuySymbol,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		CreateCloudLog(NewFmtError(err, CaptureStack()).Error(), LOG_EXCEPTION)
		return
	}
	predServClient.Post(GetLongShortEnterTradeEndpoint(pair.ID), APPLICATION_JSON, jsonBody)
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

	sellSideRes, err, debtQuantity := initPairTradeShort(bc, usdtEnterSize, pair)

	if err == nil {
		sellOrderQuoteQty, _ := strconv.ParseFloat(sellSideRes.CumulativeQuoteQty, 64)
		buySideRes, err := initPairTradeLong(bc, sellOrderQuoteQty, pair)

		if err == nil {
			updatePredServOnLongShortEnter(
				predServClient,
				pair,
				sellSideRes,
				buySideRes,
				debtQuantity,
			)
		}
	}
}

func exitLongShortTrade(
	bc *BinanceClient,
	predServClient *HttpClient,
	group *LongShortGroup,
	pair *LongShortPair,
) {
	closeLong(bc, pair)
}

func ProcessLongShortGroup(bc *BinanceClient, predServClient *HttpClient, group *LongShortGroup) {
	pairs := predServClient.FetchLongShortPairs(group.ID)

	for _, pair := range pairs {

		if pair.InPosition && !pair.IsTradeFinished && pair.ShouldClose {
			exitLongShortTrade(bc, predServClient, group, &pair)
		}

		if !pair.InPosition && !pair.IsTradeFinished && !pair.ErrorInEntering {
			// enterLongShortTrade(bc, predServClient, group, &pair)
		}
	}
}
