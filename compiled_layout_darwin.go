//go:build darwin && ane_appleneuralengine && cgo

package mlxgoane

import (
	"errors"
	"fmt"

	"github.com/tmc/apple/corefoundation"
	"github.com/tmc/apple/coregraphics"
	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/iosurface"
	"github.com/tmc/apple/objc"
	"github.com/tmc/apple/objectivec"
	"github.com/tmc/apple/private/appleneuralengine"
)

type compiledTensorLayout struct {
	Channels    int
	Width       int
	Height      int
	ElemSize    int
	RowStride   int
	PlaneStride int
	Name        string
	Symbol      string
	SymbolIndex int
}

type compiledModelSchema struct {
	Inputs  []compiledTensorLayout
	Outputs []compiledTensorLayout
	States  []compiledTensorLayout
}

func (l compiledTensorLayout) logicalCount() int {
	return l.Channels * l.Height * l.Width
}

func (l compiledTensorLayout) allocSize() int {
	return l.Channels * l.PlaneStride
}

func (l compiledTensorLayout) valid() error {
	if l.Channels <= 0 || l.Width <= 0 || l.Height <= 0 {
		return fmt.Errorf(
			"invalid tensor layout channels=%d height=%d width=%d",
			l.Channels,
			l.Height,
			l.Width,
		)
	}
	if l.ElemSize != 2 && l.ElemSize != 4 {
		return fmt.Errorf("unsupported tensor element size=%d", l.ElemSize)
	}
	if l.RowStride <= 0 || l.RowStride%64 != 0 {
		return fmt.Errorf("invalid row stride=%d", l.RowStride)
	}
	if l.PlaneStride < l.Height*l.RowStride {
		return fmt.Errorf(
			"invalid plane stride=%d want>=%d",
			l.PlaneStride,
			l.Height*l.RowStride,
		)
	}
	return nil
}

func parseCompiledModelLayouts(model objectivec.IObject) ([]compiledTensorLayout, []compiledTensorLayout, error) {
	if model.GetID() == 0 {
		return nil, nil, fmt.Errorf("compiled model is nil")
	}
	schema, err := parseCompiledModelSchema(model)
	if err != nil {
		return nil, nil, err
	}
	return schema.Inputs, schema.Outputs, nil
}

func parseCompiledModelSchema(model objectivec.IObject) (compiledModelSchema, error) {
	if model.GetID() == 0 {
		return compiledModelSchema{}, fmt.Errorf("compiled model is nil")
	}
	aneModel := appleneuralengine.ANEModelFromID(model.GetID())
	attrs := aneModel.ModelAttributes()
	if attrs.GetID() == 0 {
		return compiledModelSchema{}, fmt.Errorf("compiled model attributes are nil")
	}
	netListID := layoutDictGet(attrs, "NetworkStatusList")
	if netListID == 0 {
		return compiledModelSchema{}, fmt.Errorf("no NetworkStatusList in model attributes")
	}
	netList := foundation.NSArrayFromID(netListID)
	if netList.Count() == 0 {
		return compiledModelSchema{}, fmt.Errorf("empty NetworkStatusList")
	}
	proc := netList.ObjectAtIndex(0)

	inputSymbols, outputSymbols, err := parseProcedureSymbolMaps(attrs)
	if err != nil {
		return compiledModelSchema{}, err
	}

	inputs, err := parseCompiledTensorLayoutList(proc, "LiveInputList", inputSymbols)
	if err != nil {
		return compiledModelSchema{}, err
	}
	outputs, err := parseCompiledTensorLayoutList(proc, "LiveOutputList", outputSymbols)
	if err != nil {
		return compiledModelSchema{}, err
	}
	states, err := parseCompiledTensorLayoutList(proc, "LiveStateList", inputSymbols)
	if err != nil && !errors.Is(err, errMissingLayoutList) {
		return compiledModelSchema{}, err
	}
	if errors.Is(err, errMissingLayoutList) {
		states = nil
	}
	return compiledModelSchema{
		Inputs:  inputs,
		Outputs: outputs,
		States:  states,
	}, nil
}

var errMissingLayoutList = fmt.Errorf("missing layout list")

func parseCompiledTensorLayoutList(proc objectivec.IObject, key string, symbolIndices map[string]int) ([]compiledTensorLayout, error) {
	listID := layoutDictGet(proc, key)
	if listID == 0 {
		return nil, fmt.Errorf("%w: no %s in model attributes", errMissingLayoutList, key)
	}
	list := foundation.NSArrayFromID(listID)
	layouts := make([]compiledTensorLayout, list.Count())
	for i := uint(0); i < list.Count(); i++ {
		layout := parseCompiledTensorEntry(list.ObjectAtIndex(i), symbolIndices)
		if err := layout.valid(); err != nil {
			return nil, fmt.Errorf("%s[%d]: %w", key, i, err)
		}
		layouts[i] = layout
	}
	return layouts, nil
}

func parseCompiledTensorEntry(entry objectivec.IObject, symbolIndices map[string]int) compiledTensorLayout {
	elemSize := 2
	if layoutDictGetString(entry, "Type") == "Float32" {
		elemSize = 4
	}
	name := layoutDictGetString(entry, "Name")
	symbol := layoutDictGetString(entry, "Symbol")
	symbolIndex := -1
	if idx, ok := symbolIndices[symbol]; ok {
		symbolIndex = idx
	} else if idx, ok := symbolIndices[name]; ok {
		symbolIndex = idx
	}
	return compiledTensorLayout{
		Channels:    layoutDictGetInt(entry, "Channels"),
		Width:       layoutDictGetInt(entry, "Width"),
		Height:      layoutDictGetInt(entry, "Height"),
		ElemSize:    elemSize,
		RowStride:   layoutDictGetInt(entry, "RowStride"),
		PlaneStride: layoutDictGetInt(entry, "PlaneStride"),
		Name:        name,
		Symbol:      symbol,
		SymbolIndex: symbolIndex,
	}
}

func parseProcedureSymbolMaps(attrs foundation.INSDictionary) (map[string]int, map[string]int, error) {
	descID := layoutDictGet(attrs, "ANEFModelDescription")
	if descID == 0 {
		return nil, nil, fmt.Errorf("no ANEFModelDescription in model attributes")
	}
	desc := objectivec.ObjectFromID(descID)
	inputNames := layoutStringArray(desc, "kANEFModelInputSymbolsArrayKey")
	outputNames := layoutStringArray(desc, "kANEFModelOutputSymbolsArrayKey")
	return layoutSymbolPositions(inputNames), layoutSymbolPositions(outputNames), nil
}

func layoutStringArray(dict objectivec.IObject, key string) []string {
	id := layoutDictGet(dict, key)
	if id == 0 {
		return nil
	}
	arr := foundation.NSArrayFromID(id)
	out := make([]string, 0, arr.Count())
	for i := uint(0); i < arr.Count(); i++ {
		out = append(out, foundation.NSStringFromID(arr.ObjectAtIndex(i).GetID()).String())
	}
	return out
}

func layoutIntArray(dict objectivec.IObject, key string) []int {
	id := layoutDictGet(dict, key)
	if id == 0 {
		return nil
	}
	arr := foundation.NSArrayFromID(id)
	out := make([]int, 0, arr.Count())
	for i := uint(0); i < arr.Count(); i++ {
		out = append(out, foundation.NSNumberFromID(arr.ObjectAtIndex(i).GetID()).IntValue())
	}
	return out
}

func zipLayoutSymbolIndices(names []string, indices []int) map[string]int {
	out := make(map[string]int, len(names))
	for i, name := range names {
		if i >= len(indices) {
			break
		}
		out[name] = indices[i]
	}
	return out
}

func layoutSymbolPositions(names []string) map[string]int {
	out := make(map[string]int, len(names))
	for i, name := range names {
		out[name] = i
	}
	return out
}

func newFloatSurfaceForLayout(layout compiledTensorLayout) (coregraphics.IOSurfaceRef, error) {
	if err := layout.valid(); err != nil {
		return 0, err
	}
	props := foundation.NewMutableDictionaryWithCapacity(6)
	props.SetObjectForKey(
		foundation.NewNumberWithInt(layout.Width),
		foundation.NewStringWithString(iosurface.KIOSurfaceWidth),
	)
	props.SetObjectForKey(
		foundation.NewNumberWithInt(layout.Channels*layout.Height),
		foundation.NewStringWithString(iosurface.KIOSurfaceHeight),
	)
	props.SetObjectForKey(
		foundation.NewNumberWithInt(layout.ElemSize),
		foundation.NewStringWithString(iosurface.KIOSurfaceBytesPerElement),
	)
	props.SetObjectForKey(
		foundation.NewNumberWithInt(layout.RowStride),
		foundation.NewStringWithString(iosurface.KIOSurfaceBytesPerRow),
	)
	props.SetObjectForKey(
		foundation.NewNumberWithInt(layout.allocSize()),
		foundation.NewStringWithString(iosurface.KIOSurfaceAllocSize),
	)
	props.SetObjectForKey(
		foundation.NewNumberWithInt(0),
		foundation.NewStringWithString(iosurface.KIOSurfacePixelFormat),
	)
	raw := iosurface.IOSurfaceCreate(corefoundation.CFDictionaryRef(props.GetID()))
	if raw == 0 {
		return 0, fmt.Errorf(
			"IOSurface allocation failed for tensor [%d,%d,%d]",
			layout.Channels,
			layout.Height,
			layout.Width,
		)
	}
	return coregraphics.IOSurfaceRef(raw), nil
}

func layoutDictGet(dict objectivec.IObject, key string) objc.ID {
	return objc.Send[objc.ID](dict.GetID(), objc.Sel("objectForKey:"), objc.String(key))
}

func layoutDictGetInt(dict objectivec.IObject, key string) int {
	id := layoutDictGet(dict, key)
	if id == 0 {
		return 0
	}
	return foundation.NSNumberFromID(id).IntValue()
}

func layoutDictGetString(dict objectivec.IObject, key string) string {
	id := layoutDictGet(dict, key)
	if id == 0 {
		return ""
	}
	return foundation.NSStringFromID(id).UTF8String()
}
