#!/bin/bash

source ~/.bashrc

make clean
make runomp && {
    srun --gres=gpu:1 ./run "${MODELBIN_ROOT}/gpt-oss-20b.bin" -m getp -i data/input_test.txt -o data/output_test.txt > debug/tmp_2.txt 2> debug/tmp_err_2.txt
}

# srun --gres=gpu:1 --nodelist=MV-DZ-MI250-02 --pty bash
# rocgdb ./run
