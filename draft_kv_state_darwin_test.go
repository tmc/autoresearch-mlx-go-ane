//go:build darwin && ane_appleneuralengine && cgo

package mlxgoane

import (
	"reflect"
	"testing"
)

func TestSliceRoPERow(t *testing.T) {
	cos := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	}
	sin := []float32{
		-1, -2, -3, -4,
		-5, -6, -7, -8,
		-9, -10, -11, -12,
	}
	gotCos, gotSin, err := SliceRoPERow(cos, sin, 4, 3, 1)
	if err != nil {
		t.Fatalf("SliceRoPERow: %v", err)
	}
	if !reflect.DeepEqual(gotCos, []float32{5, 6, 7, 8}) {
		t.Fatalf("cos row=%v", gotCos)
	}
	if !reflect.DeepEqual(gotSin, []float32{-5, -6, -7, -8}) {
		t.Fatalf("sin row=%v", gotSin)
	}
}

func TestDraftKVSurfaceStateWriteAdvanceRewind(t *testing.T) {
	state, err := NewDraftKVSurfaceState(1, 2, 2, 8)
	if err != nil {
		t.Fatalf("NewDraftKVSurfaceState: %v", err)
	}
	defer state.Close()

	rowK0 := []float32{1, 2, 3, 4}
	rowV0 := []float32{5, 6, 7, 8}
	if err := state.WriteLayerKV(0, rowK0, rowV0); err != nil {
		t.Fatalf("WriteLayerKV pos0: %v", err)
	}
	if err := state.Advance(); err != nil {
		t.Fatalf("Advance pos0: %v", err)
	}
	rowK1 := []float32{9, 10, 11, 12}
	rowV1 := []float32{13, 14, 15, 16}
	if err := state.WriteLayerKV(0, rowK1, rowV1); err != nil {
		t.Fatalf("WriteLayerKV pos1: %v", err)
	}
	if err := state.Advance(); err != nil {
		t.Fatalf("Advance pos1: %v", err)
	}

	kSurf, vSurf, err := state.LayerSurfaces(0)
	if err != nil {
		t.Fatalf("LayerSurfaces: %v", err)
	}
	gotK, err := kSurf.ReadAt(0, 8)
	if err != nil {
		t.Fatalf("ReadAt K: %v", err)
	}
	gotV, err := vSurf.ReadAt(0, 8)
	if err != nil {
		t.Fatalf("ReadAt V: %v", err)
	}
	if !reflect.DeepEqual(gotK[:4], rowK0) || !reflect.DeepEqual(gotK[4:8], rowK1) {
		t.Fatalf("unexpected K rows: %v", gotK)
	}
	if !reflect.DeepEqual(gotV[:4], rowV0) || !reflect.DeepEqual(gotV[4:8], rowV1) {
		t.Fatalf("unexpected V rows: %v", gotV)
	}

	if err := state.Rewind(1); err != nil {
		t.Fatalf("Rewind: %v", err)
	}
	if got := state.Position(); got != 1 {
		t.Fatalf("position=%d want=1", got)
	}
	gotK2, err := kSurf.ReadAt(0, 8)
	if err != nil {
		t.Fatalf("ReadAt K after rewind: %v", err)
	}
	gotV2, err := vSurf.ReadAt(0, 8)
	if err != nil {
		t.Fatalf("ReadAt V after rewind: %v", err)
	}
	if !reflect.DeepEqual(gotK2[:4], rowK0) {
		t.Fatalf("K row0 changed after rewind: %v", gotK2[:4])
	}
	if !reflect.DeepEqual(gotV2[:4], rowV0) {
		t.Fatalf("V row0 changed after rewind: %v", gotV2[:4])
	}
	if !reflect.DeepEqual(gotK2[4:8], []float32{0, 0, 0, 0}) {
		t.Fatalf("K row1 not zeroed after rewind: %v", gotK2[4:8])
	}
	if !reflect.DeepEqual(gotV2[4:8], []float32{0, 0, 0, 0}) {
		t.Fatalf("V row1 not zeroed after rewind: %v", gotV2[4:8])
	}
}

func TestDraftKVSurfaceStateResetClearsWrittenPrefix(t *testing.T) {
	state, err := NewDraftKVSurfaceState(1, 2, 2, 8)
	if err != nil {
		t.Fatalf("NewDraftKVSurfaceState: %v", err)
	}
	defer state.Close()

	rowK0 := []float32{1, 2, 3, 4}
	rowV0 := []float32{5, 6, 7, 8}
	rowK1 := []float32{9, 10, 11, 12}
	rowV1 := []float32{13, 14, 15, 16}

	if err := state.WriteLayerKV(0, rowK0, rowV0); err != nil {
		t.Fatalf("WriteLayerKV pos0: %v", err)
	}
	if err := state.Advance(); err != nil {
		t.Fatalf("Advance pos0: %v", err)
	}
	if err := state.WriteLayerKV(0, rowK1, rowV1); err != nil {
		t.Fatalf("WriteLayerKV pos1: %v", err)
	}
	if err := state.Advance(); err != nil {
		t.Fatalf("Advance pos1: %v", err)
	}
	if got := state.Position(); got != 2 {
		t.Fatalf("position=%d want=2 before reset", got)
	}

	if err := state.Reset(); err != nil {
		t.Fatalf("Reset: %v", err)
	}
	if got := state.Position(); got != 0 {
		t.Fatalf("position=%d want=0 after reset", got)
	}

	kSurf, vSurf, err := state.LayerSurfaces(0)
	if err != nil {
		t.Fatalf("LayerSurfaces: %v", err)
	}
	gotK, err := kSurf.ReadAt(0, 8)
	if err != nil {
		t.Fatalf("ReadAt K after reset: %v", err)
	}
	gotV, err := vSurf.ReadAt(0, 8)
	if err != nil {
		t.Fatalf("ReadAt V after reset: %v", err)
	}
	if !reflect.DeepEqual(gotK, []float32{0, 0, 0, 0, 0, 0, 0, 0}) {
		t.Fatalf("K prefix not zeroed after reset: %v", gotK)
	}
	if !reflect.DeepEqual(gotV, []float32{0, 0, 0, 0, 0, 0, 0, 0}) {
		t.Fatalf("V prefix not zeroed after reset: %v", gotV)
	}
}
