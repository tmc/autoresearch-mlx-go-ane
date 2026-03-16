// Package decode implements the ANE decode plane engine for offloading
// per-token decode steps to the Apple Neural Engine.
//
// The engine wraps a LanguageModel and intercepts Forward() calls during
// decode (single-token) steps, routing FFN/MoE computation through ANE
// while keeping attention on GPU. Requires the ane_appleneuralengine
// build tag on Darwin.
//
// # Interface Design
//
// Consumer interfaces are defined locally per Go idiom. There are two groups:
//
// Infrastructure interfaces (stage/block/bridge) are type-asserted from the
// registered runtime backend (via exp/anehooks). These handle ANE evaluation,
// synchronization, and GPU↔ANE data transfer.
//
// Model extraction interfaces are type-asserted from models.LanguageModel.
// Models satisfy them via Go structural typing — no anehooks import needed
// on the model side. The engine asserts each interface independently; a model
// can implement a subset and the engine degrades gracefully.
//
// Weight references returned by model extraction interfaces are live (not
// copies) and valid only while the model is loaded and unmodified.
package decode
