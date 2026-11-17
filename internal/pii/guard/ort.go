package guard

import (
	"errors"
	"fmt"
	"path/filepath"
	"sync"

	"github.com/yetone/magpie/internal/pii/guard/ort"
)

// On Linux on x86-64 ONNX Runtime reads the model where its files are
// there (LoadOn): on an NVIDIA GPU through CUDA a window of 512 tokens
// takes 2.5 ms, on the CPU over Cores() threads 37, where bert.go takes
// 190 on as many (an i9-14900K, an RTX 4000 SFF Ada; #6942). The CPU
// reads the int8 model bert.go reads. CUDA has no kernels for its
// integer operators and would hand them back to the CPU, so it reads
// the publisher's fp16 model, whose windows are always 512 tokens. Both
// give each token bert.go's label, at a probability TestORTMatchesGo
// holds within .2 of bert.go's. When it can't be opened or a read
// fails, bert.go reads, as it does for CoreML.

// ORT names ONNX Runtime, as Runtime says it.
const ORT = "ONNX Runtime"

// Runtime is ONNX Runtime's files, for LoadOn.
type Runtime struct {
	// Lib is its libonnxruntime.so; "" for none, and bert.go reads.
	Lib string
	// Half is the publisher's fp16 model, to read on the GPU through
	// CUDA with the library's CUDA provider beside it; "" to read on
	// the CPU.
	Half string
}

// pad is the tokenizer's <pad>, which fills a window of the fp16 model's
// 512 after the text's tokens.
const pad = 1

// ortReader is the model in ONNX Runtime.
type ortReader struct {
	// mu lets one window in at a time: the CPU's reads over Cores()
	// threads whoever asks, and the GPU reads eight at once no faster
	// than one after another.
	mu   sync.Mutex
	s    *ort.Session
	cuda bool
	// why CUDA doesn't read, where it was to
	why error
}

// openORT opens the model in ONNX Runtime: on CUDA where rt has the fp16
// model, else, or when CUDA can't, the int8 model in dir on the CPU.
// Each is read once here, so one that can't read says so now and not at
// a request: CUDA opens cuDNN only at its first read.
func openORT(dir string, rt Runtime) (*ortReader, error) {
	var why error
	if rt.Half != "" {
		s, err := ort.Open(rt.Lib, rt.Half, 0, true)
		if err == nil {
			r := &ortReader{s: s, cuda: true}
			if _, err = r.logits([]int{0, 2}); err == nil {
				return r, nil
			}
		}
		why = fmt.Errorf("CUDA: %w", err)
	}
	s, err := ort.Open(rt.Lib, filepath.Join(dir, "model_int8.onnx"), Cores(), false)
	if err == nil {
		r := &ortReader{s: s, why: why}
		if _, err = r.logits([]int{0, 2}); err == nil {
			return r, nil
		}
	}
	if why != nil {
		return nil, fmt.Errorf("%w; on the CPU: %w", why, err)
	}
	return nil, err
}

// ortRead is (*ortReader).logits; a var for tests.
var ortRead = (*ortReader).logits

// logits are the network's scores for ids, as (*net).logits gives them.
func (r *ortReader) logits(ids []int) ([]float32, error) {
	n, size := len(ids), len(ids)
	if r.s.Len > 0 {
		if size = r.s.Len; n > size {
			return nil, errors.New("more tokens than a window")
		}
	}
	in, mask := make([]int64, size), make([]int64, size)
	for i := range in {
		in[i] = pad
	}
	for i, id := range ids {
		in[i], mask[i] = int64(id), 1
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.cuda { // its threads are cores of cores, like bert.go's
		held := take(Cores())
		defer give(held)
	}
	out, err := r.s.Run(in, mask)
	if err != nil {
		return nil, err
	}
	if len(out) != size*len(labels) {
		return nil, fmt.Errorf("ONNX Runtime gave %d scores for %d tokens", len(out), size)
	}
	return out[:n*len(labels)], nil
}

// Runtime is ONNX Runtime where it reads the model now: ORT, on "CUDA"
// or the "CPU", and then why not CUDA where LoadOn had its model. When
// it couldn't be opened, or its last read failed, bert.go reads: no
// name, and why. Nothing where LoadOn had no library.
func (m *Model) Runtime() (name, device string, err error) {
	if m.rt == nil {
		return "", "", m.RTErr
	}
	if e := m.rtFail.Load(); e != nil {
		return "", "", errors.New(*e)
	}
	if m.rt.cuda {
		return ORT, "CUDA", nil
	}
	return ORT, "CPU", m.rt.why
}
