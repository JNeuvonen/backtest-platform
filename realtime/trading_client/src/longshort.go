package main

import (
	"fmt"
)

func ProcessLongShortGroup(predServClient *HttpClient, group LongShortGroup) {
	pairs := predServClient.FetchLongShortPairs(group.ID)

	fmt.Println(pairs)
}
