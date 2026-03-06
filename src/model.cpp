#include "model.hpp"


namespace torch::nn {
    LanguageClassifier::LanguageClassifier(int vocab_size, int embed_dim, int num_classes) {
        LanguageClassifier::embedding = register_module("embedding", Embedding(vocab_size, embed_dim));
        LanguageClassifier::conv1 = register_module("conv1", Conv1d(embed_dim, 128, 3));
        LanguageClassifier::conv2 = register_module("conv2", Conv1d(128, 128, 3));
        LanguageClassifier::fc1 = register_module("fc1", Linear(128, 128));
        LanguageClassifier::fc2 = register_module("fc2", Linear(128, num_classes));
        LanguageClassifier::dropout = register_module("dropout", Dropout(0.3));
    }

    // first step - loss function & backwards pass done in train.cpp
    // takes a batch (multiple) tokenized lines of text to process in parallel.
    // layers: embedding → conv1 → conv2 → maxpool → fc1 → dropout → fc2
    torch::Tensor LanguageClassifier::forward(torch::Tensor t) {
        // shape: {batch, 200}
        t = embedding(t); // shape: {batch, 200, 32}

        // permute is a method on a tensor and takes an array of dimensions
        // needed because torch::nn::Conv1d expects {batch, channels, lengths}
        // channels describe the number of features that describe character i in the 200-length sequence.
        t = t.permute({0, 2, 1}); // shape: {batch, 32, 200}
        t = torch::relu(conv1(t)); // shape: {batch, 128, 198} - kernel size 3
        t = torch::relu(conv2(t)); // shape: {batch, 128, 196}

        // for each of the 128 features, find the maximum value across the 196 positions
        // returns a tuple (values, indices)
        t = std::get<0>(t.max(2)); // shape: {batch, 128}
        t = torch::relu(fc1(t)); // shape: {batch, 128}
        t = dropout(t);
        t = fc2(t); // shape: {batch, num_classes}, no relu because this is pure logits for cross entropy loss
        return t;
    }
}