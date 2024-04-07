package main

import (
	"runtime"
	"runtime/debug"
	"strings"
)

func GetCurrentFunctionName() string {
	pc, _, _, _ := runtime.Caller(1)
	return strings.Split(runtime.FuncForPC(pc).Name(), ".")[1]
}

func CaptureStack() string {
	return string(debug.Stack())
}
