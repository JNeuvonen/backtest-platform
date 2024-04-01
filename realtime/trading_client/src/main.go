package main

func main() {
	tradingConfig := getTradingConfig()
	NewBinanceClient(tradingConfig)
}
