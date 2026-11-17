//go:build !linux || !amd64

package ort

import "errors"

// Available says ONNX Runtime can be opened here: only on Linux on
// x86-64 (ort_linux_amd64.go).
const Available = false

var errNone = errors.New("no ONNX Runtime here")

type Session struct{ Len int }

func NVIDIA() bool { return false }

func Open(lib, model string, threads int, cuda bool) (*Session, error) { return nil, errNone }

func (*Session) Run(ids, mask []int64) ([]float32, error) { return nil, errNone }
