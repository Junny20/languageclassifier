#pragma once

#include <torch/torch.h>
#include <cstddef>
#include <string>

namespace torch::data {
    class LanguageDataset : public Dataset<LanguageDataset> {
        private:
            std::vector<std::string> texts;
            std::vector<int64_t> labels;
        public:
            explicit LanguageDataset(const std::string& csv_path);
            Example<> get(size_t index) const override;
            torch::optional<size_t> size() const override;
    };
}

"""Example is a Built in LibTorch struct"""
// struct Example {
//     Tensor data;
//     Tensor target; 
// };