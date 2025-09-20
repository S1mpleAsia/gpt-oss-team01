srun -N 1 --gres=gpu:4 ./run "/gpu_trainee/final-project/modelbin/gpt-oss-20b.bin" -m getp -i data/input.txt -o data/output.txt
# srun -N 1 --gres=gpu:4 ./run "/gpu_trainee/final-project/modelbin/gpt-oss-120b.bin" -m getp -i evaluation/input.txt -o evaluation/submission/output_120b_token_ids.txt

# srun --gres=gpu:1 --nodelist=MV-DZ-MI250-02 --pty bash
# rocgdb ./run
