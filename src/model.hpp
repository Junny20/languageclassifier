#pragma once

#include <torch/torch.h>

namespace torch::nn {
    class LanguageClassifier : public Module {
        private:
            Embedding embedding = nullptr;
            Conv1d conv1 = nullptr, conv2 = nullptr;
            Linear fc1 = nullptr, fc2 = nullptr; // fc - fully connected
            Dropout dropout = nullptr;
        public:
            explicit LanguageClassifier(int vocab_size, int embed_dim, int num_classes); // 256 by 32 by 25
            torch::Tensor forward(torch::Tensor t);
    };
}