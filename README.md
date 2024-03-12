# Magisterium 😊

```sh
docker run --rm --gpus=all \
    -v $PWD/ex:/data/ex \
    -p 8080:8080 -p 8081:8081 \
    ghcr.io/livebook-dev/livebook:latest-cuda12.1
```

```sh
docker run --rm --gpus=all -it -p 8888:8888 tensorflow/tensorflow:latest-gpu-jupyter
```

```sh
sudo chown -R $(id -u):$(id -g) .
```
