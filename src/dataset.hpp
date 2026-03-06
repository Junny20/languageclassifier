#pragma once

#include "tokenizer.hpp"
#include "csv2/reader.hpp"
#include <torch/torch.h>
#include <cstddef>
#include <string>
#include <vector>

namespace torch::data {
    class LanguageDataset : public Dataset<LanguageDataset> {
        private:
            std::vector<torch::Tensor> tokens;
            std::vector<torch::Tensor> labels;
        public:
            explicit LanguageDataset(const std::string& csv_path);
            Example<> get(size_t index) override;
            torch::optional<size_t> size() const override;
    };
}

// """Example is a Built in LibTorch struct"""
// struct Example {
//     Tensor data;
//     Tensor target; 
// };