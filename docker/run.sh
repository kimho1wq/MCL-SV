sudo docker run -it --rm --gpus all --ipc=host --shm-size 100G  --ulimit memlock=-1 --ulimit stack=67108864 \
-v /DB2:/data \
-v $PWD/exps:/exps \
-v $PWD:/workspace env202307:latest
