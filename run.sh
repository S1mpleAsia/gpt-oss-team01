srun -N 1 --gres=gpu:8 ./run "${MODELBIN_ROOT}/gpt-oss-20b.bin" -m getp -i data/input.txt -o data/output.txt

# srun --gres=gpu:1 --nodelist=MV-DZ-MI250-02 --pty bash
# rocgdb ./run
