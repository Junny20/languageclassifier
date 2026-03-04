#include <torch/torch.h>
#include <iostream>

int main()
{
    torch::Tensor t = torch::rand({3, 3});
    std::cout << t << '\n';
    return 0;
}