// Package ort reads a token classifier on ONNX Runtime, on the CPU or
// through its CUDA provider on an NVIDIA GPU. Its shared library is
// opened at run time (purego, no cgo), so a build without cgo has it
// too, and only what guard needs of its C API is here. Linux on x86-64
// (ort_linux_amd64.go); elsewhere nothing is Available.
package ort

// Version is the ONNX Runtime this was written against and tried with.
const Version = "1.32.0"
