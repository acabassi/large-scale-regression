mrun -n 1 -N 1 --cpus-per-task=36 --nodelist=node1 \
      python linear-classifier-parallel.py \
            --ps_hosts=node1:2222 \
            --worker_hosts=node2:2222,node3:2222 \
            --num_workers=2 \
            --job_name=ps \
            --task_index=0 \
            > output-parameter-server.txt &

mrun -n 1 -N 1 --cpus-per-task=36 --nodelist=node2 \
      python linear-classifier-parallel.py \
            --ps_hosts=node1:2222 \
            --worker_hosts=node2:2222,node3:2222 \
            --num_workers=2 \
            --job_name=worker \
            --task_index=0 \
            > output-first-worker.txt &

mrun -n 1 -N 1 --cpus-per-task=36 --nodelist=node3 \
      python linear-classifier-parallel.py \
            --ps_hosts=node1:2222 \
            --worker_hosts=node2:2222,node3:2222 \
            --num_workers=2 \
            --job_name=worker \
            --task_index=1 \
            > output-second-worker.txt &
