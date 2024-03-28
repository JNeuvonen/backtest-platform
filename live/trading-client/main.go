package main

import (
	"context"
	"fmt"
	"os"
)

func main() {
	binance := NewBinanceClient()
	account, err := binance.Client.NewGetAccountService().Do(context.Background())
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	fmt.Println(account)
}
