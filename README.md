# AIAssistantTest

```
Project by Mehmood Ahmad and Isaac Joffe
```
# Ricoh Theta X Setup:
```
First install (https://github.com/ricohapi/libuvc-theta)
git clone https://github.com/ricohapi/libuvc-theta.git
cd libuvc-theta
mkdir build
cd build
cmake ..
make && sudo make install

Second Install (https://github.com/nickel110/gstthetauvc)
git clone https://github.com/nickel110/gstthetauvc.git
cd gstthetauvc/thetauvc/
make
sudo cp gstthetauvc.so /usr/lib/aarch64-linux-gnu/gstreamer-1.0/gstthetauvc.so

Third Install
git clone https://github.com/umlaeute/v4l2loopback.git
cd v4l2loopback
make
make clean
make & sudo make install
sudo depmod -a

To get the camera running on dev/video1:
sudo modprobe v4l2loopback devices=1 video_nr=1 card_label="VirtualCam"
gst-launch-1.0 thetauvcsrc mode=4K ! h264parse ! avdec_h264 ! videoconvert ! v4l2sink device=/dev/video1
```
# Torch and Cuda Setup (Specific for this Project):
```
Install all the other requirements -- do this first because this can overwrite previous torch installations
download these two whl files
install these two files (pip install <<filename>>)
```
