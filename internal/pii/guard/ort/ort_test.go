//go:build linux && amd64

package ort

import (
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// files are ONNX Runtime's library and AstrLink Guard's int8 model, from
// MAGPIE_ORT_LIB (its libonnxruntime.so) and MAGPIE_GUARD_DIR (the
// model's folder): the tests that open them run only then, as no test
// downloads them.
func files(t *testing.T) (lib, model string) {
	lib, dir := os.Getenv("MAGPIE_ORT_LIB"), os.Getenv("MAGPIE_GUARD_DIR")
	if lib == "" || dir == "" {
		t.Skip("MAGPIE_ORT_LIB and MAGPIE_GUARD_DIR are not both set")
	}
	return lib, filepath.Join(dir, "model_int8.onnx")
}

// window is <s>, n-2 tokens and </s>, with its mask.
func window(n int) (ids, mask []int64) {
	ids, mask = make([]int64, n), make([]int64, n)
	for i := range ids {
		ids[i], mask[i] = int64(1000+i), 1
	}
	ids[0], ids[n-1] = 0, 2
	return ids, mask
}

// The library keeps nothing of its own on this computer: with its
// telemetry on it writes a device id and an event queue under
// ~/.cache/Microsoft as its environment is made. It reads C's
// environment, so the model is read in a process of its own, with a
// home of its own and nothing else set.
func TestKeepsNoTelemetry(t *testing.T) {
	lib, model := files(t)
	if os.Getenv("MAGPIE_ORT_CHILD") != "" {
		s, err := Open(lib, model, 1, false)
		if err != nil {
			t.Fatal(err)
		}
		ids, mask := window(12)
		out, err := s.Run(ids, mask)
		if err != nil || len(out) != 12*21 {
			t.Fatalf("read %d logits: %v", len(out), err)
		}
		t.Log("the model was read")
		return
	}
	home := t.TempDir()
	cmd := exec.Command(os.Args[0], "-test.run=^TestKeepsNoTelemetry$", "-test.v")
	cmd.Env = []string{"HOME=" + home, "PATH=" + os.Getenv("PATH"), "MAGPIE_ORT_CHILD=1",
		"MAGPIE_ORT_LIB=" + lib, "MAGPIE_GUARD_DIR=" + filepath.Dir(model)}
	out, err := cmd.CombinedOutput()
	if err != nil || !strings.Contains(string(out), "the model was read") {
		t.Fatalf("the model wasn't read: %v\n%s", err, out)
	}
	var left []string
	filepath.WalkDir(home, func(path string, d fs.DirEntry, err error) error {
		if path != home {
			left = append(left, strings.TrimPrefix(path, home))
		}
		return nil
	})
	if len(left) != 0 {
		t.Fatalf("ONNX Runtime left files in the home folder: %v", left)
	}
}

// A model is read at any length and at once; what can't be read is an
// error, not the end of magpie: a mask that isn't the window's, a model
// file cut short, and another build of the library in the same process.
func TestOpensAndRuns(t *testing.T) {
	lib, model := files(t)
	s, err := Open(lib, model, 2, false)
	if err != nil {
		t.Fatal(err)
	}
	if s.Len != 0 {
		t.Fatalf("the int8 model takes windows of %d only", s.Len)
	}
	ids, mask := window(40)
	want, err := s.Run(ids, mask)
	if err != nil || len(want) != 40*21 {
		t.Fatalf("read %d logits: %v", len(want), err)
	}
	done := make(chan string, 8)
	for range cap(done) {
		go func() {
			for range 20 {
				got, err := s.Run(ids, mask)
				if err != nil {
					done <- err.Error()
					return
				}
				for i := range got {
					if got[i] != want[i] {
						done <- "another answer for the same window"
						return
					}
				}
			}
			done <- ""
		}()
	}
	for range cap(done) {
		if e := <-done; e != "" {
			t.Fatal(e)
		}
	}
	if _, err := s.Run(ids, mask[:10]); err == nil {
		t.Fatal("a mask shorter than its window was read")
	}
	b, err := os.ReadFile(model)
	if err != nil {
		t.Fatal(err)
	}
	cut := filepath.Join(t.TempDir(), "model_int8.onnx")
	if err := os.WriteFile(cut, b[:len(b)/2], 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Open(lib, cut, 1, false); err == nil {
		t.Fatal("half a model file opened")
	}
	if _, err := Open(filepath.Join(t.TempDir(), "libonnxruntime.so"), model, 1, false); err == nil || !strings.Contains(err.Error(), "starts again") {
		t.Fatalf("another library in the same process: %v", err)
	}
}
