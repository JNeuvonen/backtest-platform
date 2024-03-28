package main

import (
	"fmt"
)

func main() {
	client := NewBinanceClient()
	res, _ := client.GetBookDepth("BTCUSDT")
	fmt.Println(res)
}
