# Useful notes about using docker

1. `docker exec -it 1b7ab9790fc7 bash` run shell inside a container

2. `docker cp wine.data 1b7ab9790fc7:/home/jovyan/wine.data` copy data from the disk into container

3. `docker run -v /Users/glebmikh/Desktop/docker:/home/jovyan/ -p 8888:8888 jupyter/scipy-notebook:2c80cf3537ca`

   `-v` means volume

   `-p` means port. First comes host port, second: docker container port

    colon indicates using a particular folder as a volume for the container

4. `docker build -t my_notebook .`

5. `docker run -v /Users/glebmikh/Desktop/docker:/home/jovyan/ -p 8888:8888 my_notebook`

6. `docker-compose up`
