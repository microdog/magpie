//go:build linux && amd64

package ort

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// Available says ONNX Runtime can be opened here.
const Available = true

// apiVersion is the ORT_API_VERSION asked for, 1.32's: an older library
// has none to give and isn't used.
const apiVersion = 32

// The members of OrtApi used, by their place in the struct
// (onnxruntime_c_api.h at v1.32.0, every member counted, the plain
// function pointers among the macros too). The struct only grows at its
// end, so a place holds for every later version; load checks the first
// three against the library before anything is made with them.
const (
	createStatus          = 0
	getErrorCode          = 1
	getErrorMessage       = 2
	createEnv             = 3
	disableTelemetry      = 6
	createSession         = 7
	run                   = 9
	createSessionOptions  = 10
	setGraphOptimization  = 23
	setIntraOpThreads     = 24
	setInterOpThreads     = 25
	sessionInputCount     = 30
	sessionInputTypeInfo  = 33
	sessionInputName      = 36
	sessionOutputName     = 37
	createTensorWithData  = 49
	tensorData            = 51
	castToTensorInfo      = 55
	tensorElementType     = 60
	dimensionsCount       = 61
	dimensions            = 62
	shapeElementCount     = 64
	tensorTypeAndShape    = 65
	createCPUMemoryInfo   = 69
	allocatorFree         = 76
	defaultAllocator      = 78
	releaseStatus         = 93
	releaseMemoryInfo     = 94
	releaseSession        = 95
	releaseValue          = 96
	releaseTypeInfo       = 98
	releaseTensorInfo     = 99
	releaseSessionOptions = 100
	appendCUDA            = 204
	createCUDAOptions     = 205
	releaseCUDAOptions    = 208
	members               = releaseCUDAOptions + 1
)

// Values of the API's enums.
const (
	invalidArgument = 2  // ORT_INVALID_ARGUMENT
	logWarning      = 2  // ORT_LOGGING_LEVEL_WARNING
	optimizeAll     = 99 // ORT_ENABLE_ALL
	arenaAllocator  = 1  // OrtArenaAllocator
	float32s        = 1  // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
	int64s          = 7  // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
)

// lib is the library, opened once a process: fn and env are set before
// path is, under mu, and only read after.
var lib struct {
	mu     sync.Mutex
	mapped string // the file dlopen took, whether or not the rest worked
	path   string // set once its API and environment are there
	fn     [members]uintptr
	env    uintptr
}

// load opens the library at path and makes its environment. Two builds
// of it (the CPU's, CUDA's) can't share a process, so once one is mapped
// another path is an error until magpie starts again.
//
// Its telemetry is switched off first, by the variable it reads as its
// environment is made and then by its own call: the 1.32 builds keep a
// device id and a queue of events under ~/.cache/Microsoft, to send to
// Microsoft, and this reads what magpie masks. A library that can't be
// told so isn't opened (TestKeepsNoTelemetry).
func load(path string) error {
	lib.mu.Lock()
	defer lib.mu.Unlock()
	if lib.path == path {
		return nil
	}
	if lib.mapped != "" && lib.mapped != path {
		return fmt.Errorf("ONNX Runtime is open from %s; %s is read once magpie starts again", lib.mapped, path)
	}
	if err := setenv("ORT_DISABLE_TELEMETRY", "1"); err != nil {
		return fmt.Errorf("ONNX Runtime's telemetry can't be switched off, so it isn't opened: %w", err)
	}
	h, err := purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_LOCAL)
	if err != nil {
		return err
	}
	lib.mapped = path
	sym, err := purego.Dlsym(h, "OrtGetApiBase")
	if err != nil {
		return err
	}
	base, _, _ := purego.SyscallN(sym)
	if base == 0 {
		return errors.New("ONNX Runtime gave no API")
	}
	// OrtApiBase is two function pointers: GetApi, GetVersionString
	api, _, _ := purego.SyscallN(*(*uintptr)(cptr(base)), apiVersion)
	if api == 0 {
		return fmt.Errorf("%s is older than ONNX Runtime %s", path, Version)
	}
	copy(lib.fn[:], unsafe.Slice((*uintptr)(cptr(api)), members))
	msg := cstr("magpie")
	st := raw(createStatus, invalidArgument, uintptr(unsafe.Pointer(&msg[0])))
	if st == 0 || gostr(raw(getErrorMessage, st)) != "magpie" || int32(raw(getErrorCode, st)) != invalidArgument {
		return fmt.Errorf("%s isn't the ONNX Runtime API this was written for", path)
	}
	raw(releaseStatus, st)
	var env uintptr
	if err := call(createEnv, logWarning, uintptr(unsafe.Pointer(&msg[0])), uintptr(unsafe.Pointer(&env))); err != nil {
		return err
	}
	if err := call(disableTelemetry, env); err != nil {
		return err
	}
	lib.env, lib.path = env, path
	return nil
}

// setenv sets a variable in C's environment alone, where the library
// reads it: os.Setenv would hand it on to every agent and program
// magpie starts after.
func setenv(key, value string) error {
	libc, err := purego.Dlopen("libc.so.6", purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return err
	}
	fn, err := purego.Dlsym(libc, "setenv")
	if err != nil {
		return err
	}
	k, v := cstr(key), cstr(value)
	if r, _, _ := purego.SyscallN(fn, uintptr(unsafe.Pointer(&k[0])), uintptr(unsafe.Pointer(&v[0])), 1); int32(r) != 0 {
		return errors.New("setenv failed")
	}
	return nil
}

// raw calls a member of the API. A pointer converted in its arguments is
// kept where it is and alive until it returns (uintptrescapes), which a
// uintptr made beforehand is not: the stack it points into can move.
//
//go:uintptrescapes
func raw(member int, args ...uintptr) uintptr {
	r, _, _ := purego.SyscallN(lib.fn[member], args...)
	return r
}

// call is raw for a member that answers with an OrtStatus: nil for
// none, else its message.
//
//go:uintptrescapes
func call(member int, args ...uintptr) error {
	st, _, _ := purego.SyscallN(lib.fn[member], args...)
	if st == 0 {
		return nil
	}
	msg := gostr(raw(getErrorMessage, st))
	raw(releaseStatus, st)
	return errors.New(msg)
}

// cptr is an address C gave, as a pointer: C's memory, which Go's
// collector neither moves nor frees.
func cptr(a uintptr) unsafe.Pointer { return *(*unsafe.Pointer)(unsafe.Pointer(&a)) }

func gostr(a uintptr) string {
	if a == 0 {
		return ""
	}
	n := 0
	for *(*byte)(unsafe.Add(cptr(a), n)) != 0 {
		n++
	}
	return string(unsafe.Slice((*byte)(cptr(a)), n))
}

func cstr(s string) []byte { return append([]byte(s), 0) }

// NVIDIA says an NVIDIA driver is loaded, so CUDA may read here.
func NVIDIA() bool {
	_, err := os.Stat("/proc/driver/nvidia/version")
	return err == nil
}

// cudaLibs are what the CUDA provider needs besides the driver's own
// libcuda: CUDA 13's runtime, named by the provider's file, and cuDNN 9,
// which it opens by name at the first read.
var cudaLibs = []string{"libcudart.so.13", "libcublasLt.so.13", "libcublas.so.13", "libcurand.so.10", "libcudnn.so.9"}

// cudaDirs are where CUDA's installers put them when the loader's own
// search (ld.so.cache, LD_LIBRARY_PATH as magpie was started) has none.
var cudaDirs = []string{"/usr/local/cuda*/lib64", "/usr/local/cuda*/targets/x86_64-linux/lib", "/opt/cuda/lib64", "/opt/cuda/targets/x86_64-linux/lib"}

var cudaReady struct {
	mu sync.Mutex
	ok bool
}

// openCUDA opens cudaLibs for the whole process, so the provider finds
// them by name wherever they are: magpie is running and can't be given
// an LD_LIBRARY_PATH now. One that isn't on this computer is said by
// name, and looked for again the next time.
func openCUDA() error {
	cudaReady.mu.Lock()
	defer cudaReady.mu.Unlock()
	if cudaReady.ok {
		return nil
	}
	for _, name := range cudaLibs {
		paths := []string{name}
		for _, d := range cudaDirs {
			found, _ := filepath.Glob(filepath.Join(d, name))
			slices.Reverse(found) // the newest of several first
			paths = append(paths, found...)
		}
		ok := false
		for _, p := range paths {
			if _, err := purego.Dlopen(p, purego.RTLD_NOW|purego.RTLD_GLOBAL); err == nil {
				ok = true
				break
			}
		}
		if !ok {
			return fmt.Errorf("%s isn't on this computer: ONNX Runtime's CUDA needs CUDA 13 and cuDNN 9", name)
		}
	}
	cudaReady.ok = true
	return nil
}

// Session is a model, loaded.
type Session struct {
	// Len is how many tokens a window has where the model takes only
	// that many (the publisher's fp16 model, 512); 0 for any.
	Len int

	h     handles
	names [][]byte  // the names below, as C reads them
	in    []uintptr // each input's name
	feed  []int     // what each input is fed: ids, the mask, zeros
	out   uintptr   // the output's name
}

type handles struct{ session, memory uintptr }

// feeds are the inputs a token classifier has, in Run's order of them.
var feeds = []string{"input_ids", "attention_mask", "token_type_ids"}

// Open loads the model at path with the library at lib: on the first
// NVIDIA GPU with cuda, else on the CPU over that many threads. The
// session is let go when nothing holds it any more.
func Open(libPath, path string, threads int, cuda bool) (*Session, error) {
	if err := load(libPath); err != nil {
		return nil, err
	}
	var opts uintptr
	if err := call(createSessionOptions, uintptr(unsafe.Pointer(&opts))); err != nil {
		return nil, err
	}
	defer raw(releaseSessionOptions, opts)
	if err := call(setGraphOptimization, opts, optimizeAll); err != nil {
		return nil, err
	}
	if cuda {
		if err := openCUDA(); err != nil {
			return nil, err
		}
		var co uintptr
		if err := call(createCUDAOptions, uintptr(unsafe.Pointer(&co))); err != nil {
			return nil, err
		}
		defer raw(releaseCUDAOptions, co)
		if err := call(appendCUDA, opts, co); err != nil {
			return nil, err
		}
	} else {
		if err := call(setIntraOpThreads, opts, uintptr(max(1, threads))); err != nil {
			return nil, err
		}
		if err := call(setInterOpThreads, opts, 1); err != nil {
			return nil, err
		}
	}
	file := cstr(path)
	var h handles
	if err := call(createSession, lib.env, uintptr(unsafe.Pointer(&file[0])), opts, uintptr(unsafe.Pointer(&h.session))); err != nil {
		return nil, err
	}
	if err := call(createCPUMemoryInfo, arenaAllocator, 0, uintptr(unsafe.Pointer(&h.memory))); err != nil {
		raw(releaseSession, h.session)
		return nil, err
	}
	s := &Session{h: h}
	runtime.AddCleanup(s, func(h handles) {
		raw(releaseSession, h.session)
		raw(releaseMemoryInfo, h.memory)
	}, h)
	if err := s.describe(); err != nil {
		return nil, err
	}
	return s, nil
}

// describe reads the model's inputs and output: each input one of feeds,
// of int64, [1, tokens].
func (s *Session) describe() error {
	var alloc, n uintptr
	if err := call(defaultAllocator, uintptr(unsafe.Pointer(&alloc))); err != nil {
		return err
	}
	if err := call(sessionInputCount, s.h.session, uintptr(unsafe.Pointer(&n))); err != nil {
		return err
	}
	name := func(member int, i uintptr) (string, error) {
		var p uintptr
		if err := call(member, s.h.session, i, alloc, uintptr(unsafe.Pointer(&p))); err != nil {
			return "", err
		}
		defer raw(allocatorFree, alloc, p)
		return gostr(p), nil
	}
	for i := range n {
		nm, err := name(sessionInputName, i)
		if err != nil {
			return err
		}
		feed := slices.Index(feeds, nm)
		if feed < 0 {
			return fmt.Errorf("the model has an input %q", nm)
		}
		kind, dims, err := s.input(i)
		if err != nil {
			return err
		}
		if kind != int64s || len(dims) != 2 {
			return fmt.Errorf("the model's %s isn't int64 [1, tokens]", nm)
		}
		if nm == feeds[0] && dims[1] > 0 {
			s.Len = int(dims[1])
		}
		s.names, s.feed = append(s.names, cstr(nm)), append(s.feed, feed)
	}
	if !slices.Contains(s.feed, 0) {
		return errors.New("the model has no input_ids")
	}
	nm, err := name(sessionOutputName, 0)
	if err != nil {
		return err
	}
	s.names = append(s.names, cstr(nm))
	for i := range s.feed {
		s.in = append(s.in, uintptr(unsafe.Pointer(&s.names[i][0])))
	}
	s.out = uintptr(unsafe.Pointer(&s.names[len(s.feed)][0]))
	return nil
}

// input is the i-th input's element type and shape, -1 where a
// dimension is free.
func (s *Session) input(i uintptr) (kind uintptr, dims []int64, err error) {
	var info, tensor, n uintptr
	if err := call(sessionInputTypeInfo, s.h.session, i, uintptr(unsafe.Pointer(&info))); err != nil {
		return 0, nil, err
	}
	defer raw(releaseTypeInfo, info)
	if err := call(castToTensorInfo, info, uintptr(unsafe.Pointer(&tensor))); err != nil {
		return 0, nil, err
	}
	if tensor == 0 {
		return 0, nil, errors.New("the model's input isn't a tensor")
	}
	if err := call(tensorElementType, tensor, uintptr(unsafe.Pointer(&kind))); err != nil {
		return 0, nil, err
	}
	if err := call(dimensionsCount, tensor, uintptr(unsafe.Pointer(&n))); err != nil {
		return 0, nil, err
	}
	if n > 0 {
		dims = make([]int64, n)
		if err := call(dimensions, tensor, uintptr(unsafe.Pointer(&dims[0])), n); err != nil {
			return 0, nil, err
		}
	}
	return kind & 0xffffffff, dims, nil
}

// Run is the model's output for one window: ids, and mask 1 at each
// token and 0 at padding, both Len long where the model has one. Runs at
// once are safe.
func (s *Session) Run(ids, mask []int64) ([]float32, error) {
	n := len(ids)
	if n == 0 || len(mask) != n || s.Len > 0 && n != s.Len {
		return nil, fmt.Errorf("a window of %d tokens and a mask of %d for a model of %d", n, len(mask), s.Len)
	}
	data := [][]int64{ids, mask, make([]int64, n)}
	shape := []int64{1, int64(n)}
	vals := make([]uintptr, len(s.feed)+1) // the inputs, then the output
	defer func() {
		for _, v := range vals {
			if v != 0 {
				raw(releaseValue, v)
			}
		}
	}()
	for i, f := range s.feed {
		d := data[f]
		if err := call(createTensorWithData, s.h.memory, uintptr(unsafe.Pointer(&d[0])), uintptr(8*n),
			uintptr(unsafe.Pointer(&shape[0])), 2, int64s, uintptr(unsafe.Pointer(&vals[i]))); err != nil {
			return nil, err
		}
	}
	out := &vals[len(s.feed)]
	err := call(run, s.h.session, 0, uintptr(unsafe.Pointer(&s.in[0])), uintptr(unsafe.Pointer(&vals[0])), uintptr(len(s.feed)),
		uintptr(unsafe.Pointer(&s.out)), 1, uintptr(unsafe.Pointer(out)))
	// the tensors point into data, and the session is let go by a cleanup
	runtime.KeepAlive(data)
	runtime.KeepAlive(s)
	if err != nil {
		return nil, err
	}
	var info, kind, count, at uintptr
	if err := call(tensorTypeAndShape, *out, uintptr(unsafe.Pointer(&info))); err != nil {
		return nil, err
	}
	defer raw(releaseTensorInfo, info)
	if err := call(tensorElementType, info, uintptr(unsafe.Pointer(&kind))); err != nil {
		return nil, err
	}
	if kind&0xffffffff != float32s {
		return nil, errors.New("the model's output isn't float32")
	}
	if err := call(shapeElementCount, info, uintptr(unsafe.Pointer(&count))); err != nil {
		return nil, err
	}
	if err := call(tensorData, *out, uintptr(unsafe.Pointer(&at))); err != nil {
		return nil, err
	}
	return slices.Clone(unsafe.Slice((*float32)(cptr(at)), count)), nil
}
