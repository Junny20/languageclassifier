#include "crow_all.h"
#include "tokenizer.hpp"
#include <fstream>
#include <torch/torch.h>
#include <onnxruntime_cxx_api.h>

int main() {
    crow::SimpleApp app;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "language-classifier");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::Session session(env, "../models/model.onnx", session_options);

    std::ifstream f("../data/mappings.json");
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto mappings = crow::json::load(content);

    CROW_ROUTE(app, "/predict").methods("POST"_method)
    ([&session, &mappings](const crow::request& req){
        auto body = crow::json::load(req.body);
        std::string text = body["text"].s();
        torch::Tensor token = tokenizer::tokenize(text);

        std::vector<int64_t> input_data(token.data_ptr<int64_t>(),
                                 token.data_ptr<int64_t>() + 200);
        std::vector<int64_t> input_shape = {1, 200};

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, input_data.data(), input_data.size(),
            input_shape.data(), input_shape.size()
        );

        const char* input_names[] = {"input"};
        const char* output_names[] = {"linear_1"};
        auto outputs = session.Run(Ort::RunOptions{nullptr}, 
            input_names, &input_tensor, 1, 
            output_names, 1
        );

        float* logits = outputs[0].GetTensorMutableData<float>();

        // find top prediction
        int top_idx = std::max_element(logits, logits + 25) - logits;

        // get language code from mappings
        std::string lang_code;
        for (auto& kv : mappings["label_to_int"]) {
            if (kv.i() == top_idx) {
                lang_code = kv.key();
                break;
            }
        }

        crow::json::wvalue response;
        response["language"] = lang_code;
        response["class_index"] = top_idx;
        return response;
    });

    CROW_ROUTE(app, "/languages")([&mappings](){
        crow::json::wvalue response;
        std::vector<std::string> languages; languages.reserve(32);
        for (auto& pair: mappings["label_to_int"]) {
            languages.push_back(pair.key());
        }
        response["languages"] = languages;
        return response;
    });

    CROW_ROUTE(app, "/health")([](){
        crow::json::wvalue response;
        response["status"] = "ok";
        response["model"] = "model.onnx";
        return response;
    });

    app.port(8080).multithreaded().run();
}