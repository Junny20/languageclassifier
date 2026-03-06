#include "tokenizer.hpp"

namespace tokenizer {
    torch::Tensor tokenize(const std::string& text, size_t max_len) {
        std::vector<int64_t> indices(max_len, 0);
        for (size_t i = 0; i < text.size() && i < max_len; ++i) {
            indices[i] = static_cast<unsigned char>(text[i]);
        }

        torch::Tensor tensor = torch::tensor(indices, torch::kInt64);
        return tensor;
    }
}