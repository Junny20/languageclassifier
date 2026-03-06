#pragma once

#include <torch/torch.h>
#include <cstdint>
#include <string>

namespace tokenizer {
    torch::Tensor tokenize(const std::string& text, size_t max_len = 200);
}
