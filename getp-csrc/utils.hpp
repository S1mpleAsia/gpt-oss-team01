#pragma once

#include "tensor.hpp"

void MemCpy_Tensor(Tensor *to, Tensor *from, long long to_offset, long long from_offset, size_t num_elem, bool copy_host, bool copy_device);
void MemSet_Tensor(Tensor *in, int value, bool set_host, bool set_device);