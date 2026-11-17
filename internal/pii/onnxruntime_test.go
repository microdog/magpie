package pii

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"errors"
	"io/fs"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/yetone/magpie/internal/pii/guard"
)

// No test fetches ONNX Runtime: one that loads the real model has no
// build to read with unless it sets one of its own (ortSandbox).
func init() { ortBuildFor = func() *ortBuild { return nil } }

// tgz is an archive laid out as ONNX Runtime's release: each of lib's
// files in the lib folder of the release's own folder, beside a header
// and a link the release has too, after extra's entries as named.
func tgz(t *testing.T, lib, extra map[string][]byte) []byte {
	t.Helper()
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)
	put := func(name string, b []byte) {
		if err := tw.WriteHeader(&tar.Header{Name: name, Mode: 0o755, Size: int64(len(b)), Typeflag: tar.TypeReg}); err != nil {
			t.Fatal(err)
		}
		tw.Write(b)
	}
	put("onnxruntime-linux-x64-9.9.9/include/onnxruntime_c_api.h", []byte("// the API"))
	tw.WriteHeader(&tar.Header{Name: "onnxruntime-linux-x64-9.9.9/lib/libonnxruntime.so", Linkname: "libonnxruntime.so.9", Typeflag: tar.TypeSymlink})
	for name, b := range extra {
		put(name, b)
	}
	for name, b := range lib {
		put("onnxruntime-linux-x64-9.9.9/lib/"+name, b)
	}
	if err := errors.Join(tw.Close(), gz.Close()); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// ortSandbox is guardSandbox with a CUDA build of ONNX Runtime to
// fetch from a server of the test's own: its two libraries in an
// archive (served as given, so a test can serve a wrong one) and the
// fp16 model. The download it starts is over before the test is.
func ortSandbox(t *testing.T, archive []byte, lib map[string][]byte) (root string, served *files) {
	t.Helper()
	root = guardSandbox(t)
	half := []byte("fp16 weights")
	served = &files{m: map[string][]byte{"/ort/onnxruntime-9.9.9.tgz": archive, "/r/" + guardRev + "/fp16/half.onnx": half}}
	s := httptest.NewServer(served)
	b := &ortBuild{dir: "onnxruntime-9.9.9-cuda", asset: pinned{"onnxruntime-9.9.9.tgz", int64(len(archive)), sum(archive)}, files: map[string]int64{}, half: true}
	for name, body := range lib {
		b.files[name] = int64(len(body))
	}
	wasBase, wasHalf, wasFor := ortBase, guardHalf, ortBuildFor
	ortBase, guardBase, guardHalf = s.URL+"/ort/", s.URL+"/r/", pinned{"fp16/half.onnx", int64(len(half)), sum(half)}
	ortBuildFor = func() *ortBuild { return b }
	t.Cleanup(func() {
		odl.stop()
		waitORT(t)
		odl.mu.Lock()
		odl.err = ""
		odl.mu.Unlock()
		s.Close()
		ortBase, guardHalf, ortBuildFor = wasBase, wasHalf, wasFor
	})
	return root, served
}

// waitORT waits for the download of ONNX Runtime's files to end.
func waitORT(t *testing.T) {
	t.Helper()
	for deadline := time.Now().Add(30 * time.Second); ; time.Sleep(2 * time.Millisecond) {
		var s Status
		if odl.fill(&s); !s.Downloading {
			return
		}
		if time.Now().After(deadline) {
			t.Fatal("ONNX Runtime's download didn't end")
		}
	}
}

// under is every file under dir, by its path in it.
func under(dir string) []string {
	var out []string
	filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err == nil && !d.IsDir() {
			rel, _ := filepath.Rel(dir, p)
			out = append(out, filepath.ToSlash(rel))
		}
		return nil
	})
	slices.Sort(out)
	return out
}

// The model's first load has no ONNX Runtime and starts its download:
// the archive's libraries, each by its name alone, and the fp16 model.
// Once they are on disk the model is let go, and the next load is given
// them. Nothing else of the archive is kept, an entry that points out
// of the folder lands nowhere, and the archive itself is deleted.
func TestONNXRuntimeIsFetchedForTheNextLoad(t *testing.T) {
	lib := map[string][]byte{ortLib: []byte("the library"), "libonnxruntime_providers_cuda.so": []byte("the CUDA provider")}
	archive := tgz(t, lib, map[string][]byte{"../../libonnxruntime_providers_cuda.so": []byte("the CUDA provider"), "onnxruntime-linux-x64-9.9.9/lib/libonnxruntime_providers_tensorrt.so": []byte("not read with")})
	root, _ := ortSandbox(t, archive, lib)
	var reads atomic.Int32
	gsrv.mu.Lock()
	gsrv.m = fakeFinder{nil, &reads}
	gen := gsrv.gen
	gsrv.mu.Unlock()
	if rt := guardRuntime(); rt != (guard.Runtime{}) {
		t.Fatalf("ONNX Runtime before it is fetched: %+v", rt)
	}
	waitORT(t)
	for deadline := time.Now().Add(10 * time.Second); ; time.Sleep(2 * time.Millisecond) {
		gsrv.mu.Lock()
		m, now := gsrv.m, gsrv.gen
		gsrv.mu.Unlock()
		if m == nil && now != gen {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("the model wasn't let go once ONNX Runtime was there: %s", ortFailed())
		}
	}
	rt := guardRuntime()
	dir := filepath.Join(guardRoot(), "onnxruntime-9.9.9-cuda")
	if rt.Lib != filepath.Join(dir, ortLib) || rt.Half != filepath.Join(guardDir(), "half.onnx") {
		t.Fatalf("ONNX Runtime once fetched: %+v", rt)
	}
	if b, _ := os.ReadFile(filepath.Join(dir, "libonnxruntime_providers_cuda.so")); string(b) != "the CUDA provider" {
		t.Fatalf("the provider is %q", b)
	}
	if b, _ := os.ReadFile(rt.Half); string(b) != "fp16 weights" {
		t.Fatalf("the fp16 model is %q", b)
	}
	want := []string{"astrlink-guard/" + guardRev[:12] + "/half.onnx", "astrlink-guard/onnxruntime-9.9.9-cuda/" + ortLib, "astrlink-guard/onnxruntime-9.9.9-cuda/libonnxruntime_providers_cuda.so"}
	if got := under(root); !slices.Equal(got, want) {
		t.Fatalf("on disk: %v, want %v", got, want)
	}
	// its folder stays when the folders of revisions before go
	os.MkdirAll(filepath.Join(guardRoot(), "onnxruntime-9.9.8"), 0o755)
	removeOldGuard()
	if got := under(root); !slices.Equal(got, want) {
		t.Fatalf("after the old folders went: %v", got)
	}
	if _, err := os.Stat(filepath.Join(guardRoot(), "onnxruntime-9.9.8")); err == nil {
		t.Fatal("another version's folder is still there")
	}
}

// An archive that isn't the pinned one, or lacks a library, is not
// unpacked: nothing is kept, the model stays loaded and reading, Now
// says why, and the next load tries again.
func TestONNXRuntimeDownloadThatFailsIsSaid(t *testing.T) {
	lib := map[string][]byte{ortLib: []byte("the library"), "libonnxruntime_providers_cuda.so": []byte("the CUDA provider")}
	good := tgz(t, lib, nil)
	for name, c := range map[string]struct {
		serve []byte
		pin   bool // pinned as served: only its files are wrong
		why   string
	}{
		"another archive": {tgz(t, map[string][]byte{ortLib: []byte("a library")}, nil), false, "bytes of"},
		"no provider":     {tgz(t, map[string][]byte{ortLib: []byte("the library")}, nil), true, "1 of ONNX Runtime's 2 files"},
		"another size":    {tgz(t, map[string][]byte{ortLib: []byte("the library"), "libonnxruntime_providers_cuda.so": []byte("a provider")}, nil), true, "libonnxruntime_providers_cuda.so is 10 bytes, not 17"},
	} {
		t.Run(name, func(t *testing.T) {
			asked := good
			if c.pin {
				asked = c.serve
			}
			_, served := ortSandbox(t, asked, lib)
			asks := func() int {
				served.mu.Lock()
				defer served.mu.Unlock()
				return len(served.ranges)
			}
			served.mu.Lock()
			served.m["/ort/onnxruntime-9.9.9.tgz"] = c.serve
			served.mu.Unlock()
			var reads atomic.Int32
			gsrv.mu.Lock()
			gsrv.m = fakeFinder{nil, &reads}
			gen := gsrv.gen
			gsrv.mu.Unlock()
			guardRuntime()
			waitORT(t)
			st := guardModel{}.Now()
			if !strings.Contains(st.RuntimeError, "downloading ONNX Runtime") || !strings.Contains(st.RuntimeError, c.why) || st.GPUError != "" || st.Error != "" {
				t.Fatalf("Now = %+v", st)
			}
			gsrv.mu.Lock()
			m, now := gsrv.m, gsrv.gen
			gsrv.mu.Unlock()
			if m == nil || now != gen || !st.Running {
				t.Fatal("the model was let go for a download that failed")
			}
			before := asks()
			if rt := guardRuntime(); rt != (guard.Runtime{}) {
				t.Fatalf("ONNX Runtime after a failed download: %+v", rt)
			}
			waitORT(t)
			if asks() == before {
				t.Fatal("the next load didn't try the download again")
			}
			if _, err := os.Stat(filepath.Join(guardRoot(), "onnxruntime-9.9.9-cuda")); err == nil {
				t.Fatalf("a folder was kept of a failed download: %v", under(guardRoot()))
			}
			if !c.pin {
				return
			}
			// the release's own archive, pinned: fetched at the next load
			b := *ortBuildFor()
			b.asset = pinned{"onnxruntime-9.9.9.tgz", int64(len(good)), sum(good)}
			ortBuildFor = func() *ortBuild { return &b }
			served.mu.Lock()
			served.m["/ort/onnxruntime-9.9.9.tgz"] = good
			served.mu.Unlock()
			guardRuntime()
			waitORT(t)
			if rt := guardRuntime(); rt.Lib == "" || rt.Half == "" {
				t.Fatalf("not fetched at the next load: %s", ortFailed())
			}
			if st := (guardModel{}).Now(); st.RuntimeError != "" {
				t.Fatalf("Now still says %q", st.RuntimeError)
			}
		})
	}
}

// runtimeFinder is a fakeFinder that says whether ONNX Runtime reads
// it, as guard.Model does.
type runtimeFinder struct {
	fakeFinder
	name, device string
	err          error
}

func (f runtimeFinder) Backend() (string, string, error) { return "", "CPU", nil }

func (f runtimeFinder) Runtime() (string, string, error) { return f.name, f.device, f.err }

// Now names ONNX Runtime where it reads AstrLink Guard: on the GPU
// through CUDA, which Settings then says, on the CPU, with why not on
// the GPU when CUDA was to read, and why it doesn't read at all. None of
// it is Via or GPUError, which Settings words as CoreML's.
func TestGuardNowNamesONNXRuntime(t *testing.T) {
	for _, c := range []struct {
		name, device, err string
		gpu               bool
	}{
		{guard.ORT, "CUDA", "", true},
		{guard.ORT, "CPU", "", false},
		{guard.ORT, "CPU", "CUDA: libcudnn.so.9 isn't on this computer", false},
		{"", "", "CUDA failure 2: out of memory", false},
		{"", "", "", false},
	} {
		t.Run(c.name+c.device+c.err, func(t *testing.T) {
			guardSandbox(t)
			fakeGuardFiles(t)
			var reads atomic.Int32
			f := runtimeFinder{fakeFinder: fakeFinder{map[string]string{}, &reads}, name: c.name, device: c.device}
			if c.err != "" {
				f.err = errors.New(c.err)
			}
			guardLoad = func(string) (spanFinder, error) { return f, nil }
			if _, err := Of(AstrLinkGuard).Find(context.Background(), []string{"hi"}); err != nil {
				t.Fatal(err)
			}
			st := Of(AstrLinkGuard).Now()
			device := "CPU"
			if c.gpu {
				device = "CUDA"
			}
			if !st.Running || st.Runtime != c.name || st.GPU != c.gpu || st.Device != device || st.RuntimeError != c.err || st.Via != "" || st.GPUError != "" {
				t.Errorf("Now = %+v", st)
			}
		})
	}
}
