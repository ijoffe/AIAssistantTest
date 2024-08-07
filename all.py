import os
import threading


def camera():
    os.system("sudo modprobe v4l2loopback devices=1 video_nr=1 card_label=\"VirtualCam\"")
    os.system("gst-launch-1.0 thetauvcsrc mode=2K ! h264parse ! avdec_h264 ! videoconvert ! v4l2sink device=/dev/video1")


def stream():
    os.chdir("/home/site/Documents/rts-app-react-publisher-viewer/")
    os.system("export VITE_RTS_STREAM_NAME=AIAssistantWebStream")
    os.system("export VITE_RTS_STREAM_PUBLISHING_TOKEN=")
    os.system("export VITE_RTS_ACCOUNT_ID=X6cJur")
    os.system("yarn nx server publisher")


def assistant():
    os.chdir("/home/site/Documents/ijoffe/AIAssistantTest/")
    os.system("python3 testing.py")


def main():
  camera_thread = threading.Thread(target=camera, args=())
  camera_thread.start()
  stream_thread = threading.Thread(target=stream, args=())
  stream_thread.start()
  input("\nPress ENTER once things are set up\n\n")
  print("\nStarting script...\n")
  assistant_thread = threading.Thread(target=assistant, args=())
  assistant_thread.start()


if __name__ == "__main__":
    main()
