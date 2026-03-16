//go:build darwin && ane_appleneuralengine && cgo

package mlxgoane

import (
	"fmt"
	"sync"
)

// DraftKVSurfaceState owns per-layer persistent K/V IOSurfaces for decode.
//
// Each layer keeps one K and one V surface. Callers write one
// [numHeads*headDim] row at the current position and then Advance after a
// successful token step.
type DraftKVSurfaceState struct {
	mu sync.Mutex

	numLayers int
	numHeads  int
	headDim   int
	maxSeqLen int
	rowSize   int
	surfaceN  int
	position  int

	k []*IOSurfaceFloat32
	v []*IOSurfaceFloat32
}

// NewDraftKVSurfaceState allocates persistent K/V IOSurface pairs for all layers.
func NewDraftKVSurfaceState(numLayers, numHeads, headDim, maxSeqLen int) (*DraftKVSurfaceState, error) {
	if numLayers <= 0 || numHeads <= 0 || headDim <= 0 || maxSeqLen <= 0 {
		return nil, fmt.Errorf(
			"new draft kv state: invalid dims layers=%d heads=%d headDim=%d maxSeq=%d",
			numLayers,
			numHeads,
			headDim,
			maxSeqLen,
		)
	}
	rowSize := numHeads * headDim
	surfaceN := rowSize * maxSeqLen
	state := &DraftKVSurfaceState{
		numLayers: numLayers,
		numHeads:  numHeads,
		headDim:   headDim,
		maxSeqLen: maxSeqLen,
		rowSize:   rowSize,
		surfaceN:  surfaceN,
		k:         make([]*IOSurfaceFloat32, numLayers),
		v:         make([]*IOSurfaceFloat32, numLayers),
	}
	cleanup := func() {
		for i := range state.k {
			if state.k[i] != nil {
				state.k[i].Close()
				state.k[i] = nil
			}
			if state.v[i] != nil {
				state.v[i].Close()
				state.v[i] = nil
			}
		}
	}
	for i := 0; i < numLayers; i++ {
		ks, err := NewIOSurfaceFloat32(surfaceN)
		if err != nil {
			cleanup()
			return nil, fmt.Errorf("new draft kv state: layer %d K surface: %w", i, err)
		}
		vs, err := NewIOSurfaceFloat32(surfaceN)
		if err != nil {
			ks.Close()
			cleanup()
			return nil, fmt.Errorf("new draft kv state: layer %d V surface: %w", i, err)
		}
		if err := ks.Fill(0); err != nil {
			vs.Close()
			ks.Close()
			cleanup()
			return nil, fmt.Errorf("new draft kv state: layer %d init K: %w", i, err)
		}
		if err := vs.Fill(0); err != nil {
			vs.Close()
			ks.Close()
			cleanup()
			return nil, fmt.Errorf("new draft kv state: layer %d init V: %w", i, err)
		}
		state.k[i] = ks
		state.v[i] = vs
	}
	return state, nil
}

// NewDraftKVSurfaceStateWithLayouts allocates persistent K/V IOSurface pairs
// from compiled model layouts.
func NewDraftKVSurfaceStateWithLayouts(
	kLayouts []compiledTensorLayout,
	vLayouts []compiledTensorLayout,
) (*DraftKVSurfaceState, error) {
	if len(kLayouts) == 0 || len(kLayouts) != len(vLayouts) {
		return nil, fmt.Errorf(
			"new draft kv state: layout count mismatch k=%d v=%d",
			len(kLayouts),
			len(vLayouts),
		)
	}
	firstK := kLayouts[0]
	firstV := vLayouts[0]
	if firstK.Height <= 0 || firstV.Height <= 0 {
		return nil, fmt.Errorf(
			"new draft kv state: invalid layout heights k=%d v=%d",
			firstK.Height,
			firstV.Height,
		)
	}
	if firstK.Height != firstV.Height {
		return nil, fmt.Errorf(
			"new draft kv state: layout height mismatch k=%d v=%d",
			firstK.Height,
			firstV.Height,
		)
	}
	rowSize := firstK.Channels * firstK.Width
	if rowSize <= 0 {
		return nil, fmt.Errorf("new draft kv state: invalid row size=%d", rowSize)
	}
	state := &DraftKVSurfaceState{
		numLayers: len(kLayouts),
		numHeads:  firstK.Channels,
		headDim:   firstK.Width,
		maxSeqLen: firstK.Height,
		rowSize:   rowSize,
		surfaceN:  firstK.logicalCount(),
		k:         make([]*IOSurfaceFloat32, len(kLayouts)),
		v:         make([]*IOSurfaceFloat32, len(vLayouts)),
	}
	cleanup := func() {
		for i := range state.k {
			if state.k[i] != nil {
				state.k[i].Close()
				state.k[i] = nil
			}
			if state.v[i] != nil {
				state.v[i].Close()
				state.v[i] = nil
			}
		}
	}
	for i := range kLayouts {
		kLayout := kLayouts[i]
		vLayout := vLayouts[i]
		switch {
		case kLayout.Height != state.maxSeqLen:
			cleanup()
			return nil, fmt.Errorf(
				"new draft kv state: layer %d K height=%d want=%d",
				i,
				kLayout.Height,
				state.maxSeqLen,
			)
		case vLayout.Height != state.maxSeqLen:
			cleanup()
			return nil, fmt.Errorf(
				"new draft kv state: layer %d V height=%d want=%d",
				i,
				vLayout.Height,
				state.maxSeqLen,
			)
		case kLayout.Channels != state.numHeads || vLayout.Channels != state.numHeads:
			cleanup()
			return nil, fmt.Errorf(
				"new draft kv state: layer %d channel mismatch k=%d v=%d want=%d",
				i,
				kLayout.Channels,
				vLayout.Channels,
				state.numHeads,
			)
		case kLayout.Width != state.headDim || vLayout.Width != state.headDim:
			cleanup()
			return nil, fmt.Errorf(
				"new draft kv state: layer %d width mismatch k=%d v=%d want=%d",
				i,
				kLayout.Width,
				vLayout.Width,
				state.headDim,
			)
		}
		ks, err := newIOSurfaceFloat32WithLayout(kLayout)
		if err != nil {
			cleanup()
			return nil, fmt.Errorf("new draft kv state: layer %d K surface: %w", i, err)
		}
		vs, err := newIOSurfaceFloat32WithLayout(vLayout)
		if err != nil {
			ks.Close()
			cleanup()
			return nil, fmt.Errorf("new draft kv state: layer %d V surface: %w", i, err)
		}
		if err := ks.Fill(0); err != nil {
			vs.Close()
			ks.Close()
			cleanup()
			return nil, fmt.Errorf("new draft kv state: layer %d init K: %w", i, err)
		}
		if err := vs.Fill(0); err != nil {
			vs.Close()
			ks.Close()
			cleanup()
			return nil, fmt.Errorf("new draft kv state: layer %d init V: %w", i, err)
		}
		state.k[i] = ks
		state.v[i] = vs
	}
	return state, nil
}

// Close releases all IOSurfaces.
func (s *DraftKVSurfaceState) Close() {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := range s.k {
		if s.k[i] != nil {
			s.k[i].Close()
			s.k[i] = nil
		}
		if s.v[i] != nil {
			s.v[i].Close()
			s.v[i] = nil
		}
	}
	s.position = 0
}

// Position returns the current decode position (next write index).
func (s *DraftKVSurfaceState) Position() int {
	if s == nil {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.position
}

// MaxSeqLen returns configured maximum sequence length.
func (s *DraftKVSurfaceState) MaxSeqLen() int {
	if s == nil {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.maxSeqLen
}

// LayerSurfaces returns K/V IOSurface wrappers for one layer.
func (s *DraftKVSurfaceState) LayerSurfaces(layer int) (*IOSurfaceFloat32, *IOSurfaceFloat32, error) {
	if s == nil {
		return nil, nil, fmt.Errorf("draft kv state: state is nil")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if layer < 0 || layer >= s.numLayers {
		return nil, nil, fmt.Errorf("draft kv state: layer=%d out of range [0,%d)", layer, s.numLayers)
	}
	if s.k[layer] == nil || s.v[layer] == nil {
		return nil, nil, fmt.Errorf("draft kv state: layer=%d surfaces are unavailable", layer)
	}
	return s.k[layer], s.v[layer], nil
}

// WriteLayerKV writes one token row at the current position for a layer.
func (s *DraftKVSurfaceState) WriteLayerKV(layer int, kRow, vRow []float32) error {
	if s == nil {
		return fmt.Errorf("draft kv state: state is nil")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if layer < 0 || layer >= s.numLayers {
		return fmt.Errorf("draft kv state: layer=%d out of range [0,%d)", layer, s.numLayers)
	}
	if s.position < 0 || s.position >= s.maxSeqLen {
		return fmt.Errorf("draft kv state: position=%d out of range [0,%d)", s.position, s.maxSeqLen)
	}
	if len(kRow) != s.rowSize || len(vRow) != s.rowSize {
		return fmt.Errorf(
			"draft kv state: row len mismatch k=%d v=%d want=%d",
			len(kRow),
			len(vRow),
			s.rowSize,
		)
	}
	offset := s.position * s.rowSize
	if err := s.k[layer].WriteAt(offset, kRow); err != nil {
		return fmt.Errorf("draft kv state: layer %d write K at pos=%d: %w", layer, s.position, err)
	}
	if err := s.v[layer].WriteAt(offset, vRow); err != nil {
		return fmt.Errorf("draft kv state: layer %d write V at pos=%d: %w", layer, s.position, err)
	}
	return nil
}

// Advance increments the decode position.
func (s *DraftKVSurfaceState) Advance() error {
	if s == nil {
		return fmt.Errorf("draft kv state: state is nil")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.position+1 > s.maxSeqLen {
		return fmt.Errorf("draft kv state: advance beyond max seq len=%d", s.maxSeqLen)
	}
	s.position++
	return nil
}

// Reset clears all K/V surfaces and rewinds position to zero.
func (s *DraftKVSurfaceState) Reset() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.resetLocked()
}

// Rewind moves position backward and clears truncated rows.
func (s *DraftKVSurfaceState) Rewind(n int) error {
	if s == nil || n <= 0 {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if n >= s.position {
		return s.resetLocked()
	}
	newPos := s.position - n
	zero := make([]float32, n*s.rowSize)
	start := newPos * s.rowSize
	for i := 0; i < s.numLayers; i++ {
		if s.k[i] == nil || s.v[i] == nil {
			continue
		}
		if err := s.k[i].WriteAt(start, zero); err != nil {
			return fmt.Errorf("draft kv state: rewind layer %d K: %w", i, err)
		}
		if err := s.v[i].WriteAt(start, zero); err != nil {
			return fmt.Errorf("draft kv state: rewind layer %d V: %w", i, err)
		}
	}
	s.position = newPos
	return nil
}

func (s *DraftKVSurfaceState) resetLocked() error {
	used := s.position * s.rowSize
	if used <= 0 {
		s.position = 0
		return nil
	}
	zero := make([]float32, used)
	for i := 0; i < s.numLayers; i++ {
		if s.k[i] == nil || s.v[i] == nil {
			continue
		}
		if err := s.k[i].WriteAt(0, zero); err != nil {
			return fmt.Errorf("draft kv state: reset layer %d K: %w", i, err)
		}
		if err := s.v[i].WriteAt(0, zero); err != nil {
			return fmt.Errorf("draft kv state: reset layer %d V: %w", i, err)
		}
	}
	s.position = 0
	return nil
}

// SliceRoPERow returns one [headDim] row from flattened [maxSeqLen*headDim] tables.
func SliceRoPERow(cosTable, sinTable []float32, headDim, maxSeqLen, position int) ([]float32, []float32, error) {
	if headDim <= 0 || maxSeqLen <= 0 {
		return nil, nil, fmt.Errorf("slice rope row: invalid dims headDim=%d maxSeq=%d", headDim, maxSeqLen)
	}
	want := headDim * maxSeqLen
	if len(cosTable) != want || len(sinTable) != want {
		return nil, nil, fmt.Errorf("slice rope row: table len mismatch cos=%d sin=%d want=%d", len(cosTable), len(sinTable), want)
	}
	if position < 0 || position >= maxSeqLen {
		return nil, nil, fmt.Errorf("slice rope row: position=%d out of range [0,%d)", position, maxSeqLen)
	}
	start := position * headDim
	end := start + headDim
	cos := make([]float32, headDim)
	sin := make([]float32, headDim)
	copy(cos, cosTable[start:end])
	copy(sin, sinTable[start:end])
	return cos, sin, nil
}
