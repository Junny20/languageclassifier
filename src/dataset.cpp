#include "dataset.hpp"

namespace torch::data
{

    LanguageDataset::LanguageDataset(const std::string &csv_path)
    {
        csv2::Reader<
            csv2::delimiter<','>,
            csv2::quote_character<'"'>,
            csv2::first_row_is_header<true>,
            csv2::trim_policy::trim_whitespace 
        > reader;

        if (reader.mmap(csv_path))
        {
            for (const auto &row : reader)
            {
                std::vector<std::string> line;
                line.reserve(2);
                for (const auto &cell : row)
                {
                    std::string value;
                    cell.read_value(value);
                    line.push_back(value);
                    if (line.size() == 2)
                    {
                        tokens.push_back(tokenizer::tokenize(line[0]));
                        labels.push_back(torch::tensor(std::stoll(line[1]), torch::kInt64));
                    }
                }
            }
        }
    }

    Example<> LanguageDataset::get(size_t index)
    {
        return {tokens.at(index), labels.at(index)};
    }

    torch::optional<size_t> LanguageDataset::size() const
    {
        return tokens.size();
    }
}