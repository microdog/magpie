package guard

import (
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/yetone/magpie/internal/pii/guard/ort"
)

// ortFiles is ONNX Runtime's library, given in MAGPIE_ORT_LIB, and with
// MAGPIE_GUARD_HALF the publisher's fp16 model for its CUDA (the
// library then the CUDA build's, its provider beside it): the tests
// that read through it run only then, as no test downloads them.
func ortFiles(t *testing.T) Runtime {
	if !ort.Available {
		t.Skip("no ONNX Runtime here")
	}
	lib := os.Getenv("MAGPIE_ORT_LIB")
	if lib == "" {
		t.Skip("MAGPIE_ORT_LIB is not set")
	}
	return Runtime{Lib: lib, Half: os.Getenv("MAGPIE_GUARD_HALF")}
}

// people is n lines, each a name, a phone, an email and an address.
func people(n int) string {
	var b strings.Builder
	for i := range n {
		fmt.Fprintf(&b, "第%d位联系人：张三，电话 138%08d，邮箱 user%d@example.com，住在北京市海淀区中关村大街%d号。\n", i, i*7919, i, i)
	}
	return b.String()
}

// ONNX Runtime, on the CPU and on CUDA, gives every token the label
// bert.go gives it, at a probability within .2 of bert.go's, for a
// short window, one of Chinese and English, and a full one; and in a
// text of several windows what either is sure of (.9) the other finds
// too. A window read wrong (its padding unmasked, say) fails here.
func TestORTMatchesGo(t *testing.T) {
	d := dir(t)
	rt := ortFiles(t)
	texts := []string{
		"hi",
		"联系人：张三，电话 13800138000，邮箱 zhangsan@example.com。",
		"Send the invoice to Maria Schmidt, 221B Baker Street, London. Her card is 4111 1111 1111 1111 and the key is sk-proj-9fK2mQx7Lw3ZtB8nV5cR1yD4hJ6pA0sE.\n" +
			"服务器 10.24.8.17 的管理员是李雷，生日 1990-05-17，住在上海市浦东新区世纪大道100号。",
		people(40),
	}
	for _, on := range []string{"CPU", "CUDA"} {
		t.Run(on, func(t *testing.T) {
			use := Runtime{Lib: rt.Lib}
			if on == "CUDA" {
				if use.Half = rt.Half; use.Half == "" {
					t.Skip("MAGPIE_GUARD_HALF is not set")
				}
			}
			m, err := LoadOn(d, use)
			if err != nil {
				t.Fatal(err)
			}
			if name, dev, err := m.Runtime(); name != ORT || dev != on || err != nil {
				t.Fatalf("ONNX Runtime reads on %q %q: %v (%v)", name, dev, err, m.RTErr)
			}
			k := len(labels)
			var worst float64
			for _, text := range texts {
				toks := m.tok.encode(text)
				ids := []int{0}
				for _, tk := range toks[:min(len(toks), window)] {
					ids = append(ids, tk.id)
				}
				ids = append(ids, 2)
				want := m.net.logits(ids, 4)
				got := m.logits(ids, 1)
				if e := m.rtFail.Load(); e != nil {
					t.Fatalf("a window of %d tokens: %s", len(ids), *e)
				}
				if len(got) != len(want) {
					t.Fatalf("a window of %d tokens: %d scores, bert.go %d", len(ids), len(got), len(want))
				}
				for i := range ids {
					a, b := want[i*k:(i+1)*k], got[i*k:(i+1)*k]
					if argmax(a) != argmax(b) {
						t.Errorf("token %d of %d: %s, bert.go %s", i, len(ids), labels[argmax(b)], labels[argmax(a)])
					}
					worst = max(worst, math.Abs(top(a)-top(b)))
				}
			}
			t.Logf("probabilities within %.3f of bert.go's", worst)
			if worst > .2 {
				t.Errorf("probabilities %.3f from bert.go's", worst)
			}
			text := people(200)
			var reads atomic.Int32
			t.Cleanup(func() { ortRead = (*ortReader).logits })
			ortRead = func(r *ortReader, ids []int) ([]float32, error) {
				reads.Add(1)
				return r.logits(ids)
			}
			theirs := m.Find(text)
			if n := len(m.tok.encode(text)); int(reads.Load()) < n/window || m.rtFail.Load() != nil {
				t.Fatalf("ONNX Runtime read %d windows of a text of %d tokens (%v)", reads.Load(), n, m.rtFail.Load())
			}
			m.rt = nil
			ours := m.Find(text)
			if len(ours) < 600 {
				t.Fatalf("bert.go found %d spans", len(ours))
			}
			for _, pair := range [][2][]Span{{ours, theirs}, {theirs, ours}} {
				have := map[[2]int]string{}
				for _, s := range pair[1] {
					have[[2]int{s.Start, s.End}] = s.Label
				}
				for _, s := range pair[0] {
					if s.Score >= .9 && have[[2]int{s.Start, s.End}] != s.Label {
						t.Errorf("%s %q at %.2f is %q to the other", s.Label, text[s.Start:s.End], s.Score, have[[2]int{s.Start, s.End}])
					}
				}
			}
		})
	}
}

// top is the probability of a token's best label.
func top(row []float32) float64 {
	mx := row[argmax(row)]
	var sum float64
	for _, v := range row {
		sum += math.Exp(float64(v - mx))
	}
	return 1 / sum
}

// A read ONNX Runtime fails is read by bert.go, the same scores and the
// same spans as with no ONNX Runtime, and Runtime says why until a read
// works again.
func TestFailedORTReadFallsBack(t *testing.T) {
	m, err := Load(dir(t))
	if err != nil {
		t.Fatal(err)
	}
	m.ml = nil
	text := people(30)
	want := m.Find(text)
	ids := []int{0, 5, 6, 7, 8, 2}
	scores := m.net.logits(ids, 1)
	m.rt = &ortReader{}
	t.Cleanup(func() { ortRead = (*ortReader).logits })
	ortRead = func(*ortReader, []int) ([]float32, error) { return nil, errors.New("CUDA failure 2: out of memory") }
	if got := m.logits(ids, 1); !reflect.DeepEqual(got, scores) {
		t.Fatalf("bert.go read %d scores, not its own %d", len(got), len(scores))
	}
	if name, dev, err := m.Runtime(); name != "" || dev != "" || err == nil || err.Error() != "CUDA failure 2: out of memory" {
		t.Fatalf("after a failed read: %q %q %v", name, dev, err)
	}
	if got := m.Find(text); len(want) == 0 || !reflect.DeepEqual(got, want) {
		t.Fatalf("found %d spans with every read failing, %d without ONNX Runtime", len(got), len(want))
	}
	ortRead = func(*ortReader, []int) ([]float32, error) { return scores, nil }
	m.logits(ids, 1)
	if name, dev, err := m.Runtime(); name != ORT || dev != "CPU" || err != nil {
		t.Fatalf("after a read that worked: %q %q %v", name, dev, err)
	}
}

// A library that can't be opened leaves the model loaded and bert.go
// reading, with why.
func TestORTThatCannotOpenLeavesGo(t *testing.T) {
	m, err := LoadOn(dir(t), Runtime{Lib: filepath.Join(t.TempDir(), "libonnxruntime.so.1.32.0")})
	if err != nil {
		t.Fatal(err)
	}
	if name, dev, err := m.Runtime(); m.rt != nil || name != "" || dev != "" || err == nil {
		t.Fatalf("no library: %q %q %v", name, dev, err)
	}
	if m.ml == nil && len(m.Find("联系人：张三，电话 13800138000。")) == 0 {
		t.Fatal("bert.go found nothing")
	}
}

// Runtime names what reads: nothing where there is no ONNX Runtime, it
// on CUDA, it on the CPU with why CUDA doesn't, and why it doesn't
// read at all.
func TestRuntime(t *testing.T) {
	if name, dev, err := (&Model{}).Runtime(); name != "" || dev != "" || err != nil {
		t.Fatalf("none: %q %q %v", name, dev, err)
	}
	opened := errors.New("no such file")
	if name, dev, err := (&Model{RTErr: opened}).Runtime(); name != "" || dev != "" || err != opened {
		t.Fatalf("not opened: %q %q %v", name, dev, err)
	}
	if name, dev, err := (&Model{rt: &ortReader{cuda: true}}).Runtime(); name != ORT || dev != "CUDA" || err != nil {
		t.Fatalf("CUDA: %q %q %v", name, dev, err)
	}
	why := errors.New("CUDA: libcudnn.so.9 isn't on this computer")
	if name, dev, err := (&Model{rt: &ortReader{why: why}}).Runtime(); name != ORT || dev != "CPU" || err != why {
		t.Fatalf("the CPU: %q %q %v", name, dev, err)
	}
}
