#pragma once

#include <torch/torch.h>

namespace torch::nn {
    class LanguageClassifier : public Module {
        private:
            Embedding embedding;
            Conv1d conv1, conv2;
            Linear fc1, fc2; // fc - fully connected
            Dropout dropout;
        public:
            explicit LanguageClassifier(int vocab_size, int embed_dim, int num_classes); // 256 by 32 by 25
            torch::Tensor forward(torch::Tensor t);
    };
}