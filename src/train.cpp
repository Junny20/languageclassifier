#include "model.hpp"
#include "dataset.hpp"
#include <torch/torch.h>
#include <iostream>

int main()
{
    torch::data::LanguageDataset training_data = torch::data::LanguageDataset("../data/training.csv");
    torch::data::LanguageDataset validation_data = torch::data::LanguageDataset("../data/validation.csv");
    std::shared_ptr<torch::nn::LanguageClassifier> model = std::make_shared<torch::nn::LanguageClassifier>(256, 32, 25);
    
    auto training_loader = torch::data::make_data_loader(
        std::move(training_data).map(torch::data::transforms::Stack<>()), // stack individual samples into batched tensors
        torch::data::DataLoaderOptions().batch_size(32).workers(2)
    );

    auto validation_loader = torch::data::make_data_loader(
        std::move(validation_data).map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions().batch_size(32).workers(2)
    );

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));
    torch::nn::CrossEntropyLoss criterion;

    unsigned int epochs = 10;
    for (int e = 0; e < epochs; ++e) {
        model->train();
        double running_loss = 0.0;
        int batch_count = 0;

        for (auto& batch: *training_loader) {
            torch::Tensor input = batch.data; // shape: {batch_size, 200}
            // squeeze changes dims from {batch_size, 1} to {batch_size}.
            torch::Tensor labels = batch.target.squeeze(); // shape: {batch_size}

            optimizer.zero_grad();

            torch::Tensor predictions = model->forward(input); // forward pass
            torch::Tensor loss = criterion(predictions, labels); // loss function
            loss.backward(); // backprop
            optimizer.step(); // update weights

            running_loss += loss.item<double>();
            ++batch_count;
        }

        model->eval();
        torch::NoGradGuard no_grad; // autograd disabled, saves memory
        unsigned int correct = 0, total = 0;

        for (auto& batch: *validation_loader) {
            torch::Tensor input = batch.data;
            torch::Tensor labels = batch.target.squeeze();

            torch::Tensor predictions = model->forward(input);
            torch::Tensor predicted = predictions.argmax(1);
            correct += predicted.eq(labels).sum().item<int>();
            total += labels.size(0);
        }

        double avg_loss = running_loss / batch_count;
        double accuracy = 100.0 * correct / total;
        std::cout << "Epoch: " << e + 1 << " | Loss: " << avg_loss << " | Validation Accuracy: " << accuracy << "%\n";
    }

    torch::save(model, "model.pt");
    std::cout << "Model saved to model.pt\n";    
}