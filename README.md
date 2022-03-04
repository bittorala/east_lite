```
docker build . -t east
```
```
docker run --name east -d -v /home/bittor/data/:/data/ --gpus all -it east bash
```
