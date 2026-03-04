## Prerequisites

**LibTorch**
Download the pre-built LibTorch binaries for your platform from [pytorch.org](https://pytorch.org/cppdocs/installing.html):
- Mac (Apple Silicon): `libtorch-macos-arm64-2.10.0.zip`
- Linux (CPU): `libtorch-cxx11-abi-shared-with-deps-2.10.0+cpu.zip`

Unzip to your home directory:
```bash
unzip libtorch-*.zip -d ~/
```

**libomp (Mac only)**
```bash
brew install libomp
```
```