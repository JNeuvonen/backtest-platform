package main

import (
	"runtime"
	"strings"
)

func GetCurrentFunctionName() string {
	pc, _, _, _ := runtime.Caller(1)
	return strings.Split(runtime.FuncForPC(pc).Name(), ".")[1]
}
