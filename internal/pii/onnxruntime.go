package pii

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"sync"
	"sync/atomic"

	"github.com/yetone/magpie/internal/pii/guard"
	"github.com/yetone/magpie/internal/pii/guard/ort"
)

// On Linux on x86-64 ONNX Runtime reads AstrLink Guard (guard.LoadOn):
// its CPU build about five times as fast as bert.go, and on a computer
// with an NVIDIA driver its CUDA build on the GPU, with the publisher's
// fp16 model (#6942). Its files are not the model's: the model is on
// disk and reads without them, on bert.go, so they are fetched after it
// loads, in the background, and the model is let go once they are there
// so that the next request loads it with them. They are the release's
// archive from ONNX Runtime's own GitHub, pinned at the sha256 GitHub
// lists for it, of which only the libraries named are kept, and the
// fp16 model at guardRev, pinned at the sha256 Hugging Face lists.
// Size doesn't count them.

// ortBuild is a build of ONNX Runtime's release.
type ortBuild struct {
	dir   string           // its folder in guardRoot
	asset pinned           // the release's archive
	files map[string]int64 // what is kept of the archive's lib folder, by size
	half  bool             // reads the fp16 model, on CUDA
}

// ortLib is the library's file in a build's folder.
const ortLib = "libonnxruntime.so." + ort.Version

var ortBase = "https://github.com/microsoft/onnxruntime/releases/download/v" + ort.Version + "/"

var ortCPU = &ortBuild{
	dir:   "onnxruntime-" + ort.Version,
	asset: pinned{"onnxruntime-linux-x64-" + ort.Version + ".tgz", 13425244, "bb169b7e374731730d2ae406fb39de7be4b315f413a9132efe896261c31bb503"},
	files: map[string]int64{ortLib: 34824680},
}

// ortCUDA is the build for CUDA 13, the one this driver's CUDA is; its
// library reads on the CPU too, where CUDA can't.
var ortCUDA = &ortBuild{
	dir:   "onnxruntime-" + ort.Version + "-cuda13",
	asset: pinned{"onnxruntime-linux-x64-gpu_cuda13-" + ort.Version + ".tgz", 245630872, "f7f4bdaa2db5edd0652e9c205f2cba18317f74af3bf2fdeefa34b5af160f10f0"},
	files: map[string]int64{ortLib: 38068736, "libonnxruntime_providers_shared.so": 14632, "libonnxruntime_providers_cuda.so": 289454752},
	half:  true,
}

// guardHalf is the publisher's fp16 model, at guardRev, kept beside the
// int8 one.
var guardHalf = pinned{"fp16/model_fp16_static512.onnx", 73945956, "ade35d3d5dbec10860d3cea5b25ad0facb27fa59c0e6547d91598afaa0012af6"}

// ortBuildFor is the build this computer reads with: CUDA's where an
// NVIDIA driver is loaded, else the CPU's, and none where ONNX Runtime
// doesn't run. It is asked once, since a process keeps the build it
// opened. A var for tests.
var ortBuildFor = sync.OnceValue(func() *ortBuild {
	switch {
	case !ort.Available:
		return nil
	case ort.NVIDIA():
		return ortCUDA
	}
	return ortCPU
})

func ortDir(b *ortBuild) string { return filepath.Join(guardRoot(), b.dir) }

func halfPath() string { return filepath.Join(guardDir(), path.Base(guardHalf.path)) }

// ortInstalled says every file b reads with is on disk at its size.
func ortInstalled(b *ortBuild) bool {
	for name, size := range b.files {
		if fi, err := os.Stat(filepath.Join(ortDir(b), name)); err != nil || fi.Size() != size {
			return false
		}
	}
	if b.half {
		if fi, err := os.Stat(halfPath()); err != nil || fi.Size() != guardHalf.size {
			return false
		}
	}
	return true
}

// odl is the download of ONNX Runtime's files.
var odl dlState

// guardRuntime is ONNX Runtime's files for guard.LoadOn, once they are
// on disk; until then none, and their download is started unless it is
// under way. A download that failed is tried again at the model's next
// load.
func guardRuntime() guard.Runtime {
	b := ortBuildFor()
	if b == nil {
		return guard.Runtime{}
	}
	if !ortInstalled(b) {
		odl.run(0, ortInstall, guardUnload)
		return guard.Runtime{}
	}
	rt := guard.Runtime{Lib: filepath.Join(ortDir(b), ortLib)}
	if b.half {
		rt.Half = halfPath()
	}
	return rt
}

// ortInstall fetches what isn't on disk of the build's library and, for
// CUDA, the fp16 model.
func ortInstall(ctx context.Context) error {
	b := ortBuildFor()
	if b == nil {
		return nil
	}
	var n atomic.Int64
	if !ortInstalled(&ortBuild{dir: b.dir, files: b.files}) {
		if err := ortUnpack(ctx, b, &n); err != nil {
			return fmt.Errorf("downloading ONNX Runtime: %w", err)
		}
	}
	if b.half && !ortInstalled(b) {
		if err := os.MkdirAll(guardDir(), 0o755); err != nil {
			return err
		}
		if err := fetch(ctx, guardURL(guardHalf), halfPath(), guardHalf.size, guardHalf.sha, &n); err != nil {
			return fmt.Errorf("downloading %s: %w", path.Base(guardHalf.path), err)
		}
	}
	return nil
}

// ortUnpack fetches b's archive and puts its files in b's folder, which
// is there only once every one of them is.
func ortUnpack(ctx context.Context, b *ortBuild, n *atomic.Int64) error {
	if err := os.MkdirAll(guardRoot(), 0o755); err != nil {
		return err
	}
	tgz := ortDir(b) + ".tgz"
	if err := fetch(ctx, ortBase+b.asset.path, tgz, b.asset.size, b.asset.sha, n); err != nil {
		return err
	}
	defer os.Remove(tgz)
	tmp := ortDir(b) + ".part"
	os.RemoveAll(tmp)
	if err := os.MkdirAll(tmp, 0o755); err != nil {
		return err
	}
	if err := untar(tgz, tmp, b.files); err != nil {
		os.RemoveAll(tmp)
		return err
	}
	os.RemoveAll(ortDir(b))
	return os.Rename(tmp, ortDir(b))
}

// untar puts the archive's files named in want in dir, each by its name
// alone and only at the size want gives it: nothing else of the archive
// is kept, and nothing lands outside dir.
func untar(tgz, dir string, want map[string]int64) error {
	f, err := os.Open(tgz)
	if err != nil {
		return err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	tr := tar.NewReader(gz)
	got := map[string]bool{}
	for {
		h, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		name := path.Base(h.Name)
		size, ok := want[name]
		if !ok || h.Typeflag != tar.TypeReg || got[name] {
			continue
		}
		if h.Size != size {
			return fmt.Errorf("the archive's %s is %d bytes, not %d", name, h.Size, size)
		}
		out, err := os.OpenFile(filepath.Join(dir, name), os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o755)
		if err != nil {
			return err
		}
		if _, err := io.Copy(out, tr); err != nil {
			out.Close()
			return err
		}
		if err := out.Close(); err != nil {
			return err
		}
		got[name] = true
	}
	if len(got) != len(want) {
		return fmt.Errorf("the archive has %d of ONNX Runtime's %d files", len(got), len(want))
	}
	return nil
}

// keptInRoot says a folder in guardRoot is this revision's model or
// this computer's build of ONNX Runtime (or that build being unpacked),
// which removeOldGuard leaves.
func keptInRoot(name string) bool {
	if name == filepath.Base(guardDir()) {
		return true
	}
	b := ortBuildFor()
	return b != nil && (name == b.dir || name == b.dir+".part")
}

// ortFailed is why ONNX Runtime's last download stopped, "" when it
// worked or none ran.
func ortFailed() string {
	var s Status
	odl.fill(&s)
	return s.Error
}

// runtimer is a model that says whether ONNX Runtime reads it
// (guard.Model.Runtime).
type runtimer interface {
	Runtime() (name, device string, err error)
}
