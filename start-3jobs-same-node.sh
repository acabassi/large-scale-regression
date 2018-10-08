mrun -n 1 -N 1 --cpus-per-task=5 --shared --nodelist=node1 \
      python linear-classifier-parallel.py \
            --ps_hosts=node1:2222 \
            --worker_hosts=node1:2223,node1:2224 \
            --num_workers=2 \
            --job_name=ps \
            --task_index=0 \
            > output-parameter-server.txt &

mrun -n 1 -N 1 --cpus-per-task=5 --shared --nodelist=node1 \
      python linear-classifier-parallel.py \
            --ps_hosts=node1:2222 \
            --worker_hosts=node1:2223,node1:2224 \
            --num_workers=2 \
            --job_name=worker \
            --task_index=0 \
            > output-first-worker.txt &

mrun -n 1 -N 1 --cpus-per-task=5 --shared --nodelist=node1 \
      python linear-classifier-parallel.py \
            --ps_hosts=node1:2222 \
            --worker_hosts=node1:2223,node1:2224 \
            --num_workers=2 \
            --job_name=worker \
            --task_index=0 \
            > output-second-worker.txt &
