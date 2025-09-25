#pragma once

// Bench-local stub to satisfy includes of config.hpp from library sources.
// Provides forward declarations and extern pointers used by headers.

struct Config;
struct Transformer;
struct Tokenizer;
struct Sampler;
struct Requests;

extern Config *public_config;
extern Transformer *public_transformer;
extern Tokenizer *public_tokenizer;
extern Sampler *public_sampler;
extern Requests *public_requests;
