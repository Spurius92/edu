# My notes on how to setup <https://github.com/alexschultz/deepracer-for-dummies> in Amazon ec2 instance

1. Create instance, don't forget to get 30 gb of ssd volume

2. download .pem file, put it in .aws folder, do `chmod 400 pk.pem` on this file

3. `ssh -i .aws/pk.pem ubuntu@ec2-xxx-xxx-xxx-xxx.compute-1.amazonaws.com`

    `ubuntu` is a defaukt user name in the ec2 instance
    `ec2...` is a public dns address of the instance

4. `scp -i .aws\new.pem E:\VNC-Viewer-6.19.923-Linux-x64.deb ubuntu@ec2-3-82-128-213.compute-1.amazonaws.com:`

5. run `./setup.sh` inside the ec2 machine
