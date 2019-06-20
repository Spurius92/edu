# Here is how i tried to install Nvidia driver for Ubuntu server 18.04 on the old desktop

## And in the end the it turns out, that my Geforce 9600 GT doesn't support CUDA, which means, no need to install driver at all 

1. Check the version of device `sudo lspci -vnn | grep -i VGA -A 12`
  It showed Nvidia Geforce 9600 GT as expected

2. `sudo apt install ubuntu-drivers-common` and `ubuntu-drivers devices`
  It showed the available options for installation: only nvidia-340, which is good

3. `sudo add-apt-repository ppa:graphics-drivers/ppa` added repository for graphics drivers
  It didn't help in my case

4. `sudo apt install nvidia-340`

5. `sudo nvidia-xconfig --no-use-edid-dpi`

6. `sudo nano /etc/X11/xorg.conf` and add in the Monitor section `Option "DPI" "96 x 96"`
