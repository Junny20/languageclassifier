#pragma once

#include <tokenizer.hpp>
#include <torch/torch.h>
#include <string>

namespace tokenizer {
    torch::Tensor tokenize(const std::string& text, size_t max_len = 200);
}