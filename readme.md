

## Quick Start



```shell
# download yolov3 weights from YOLO website
wget -O weights/yolov3.weights https://pjreddie.com/media/files/yolov3.weights
wget -O weights/yolov3-spp.weights https://pjreddie.com/media/files/yolov3-spp.weights

# convert weights to .h5 file
python tools/convert.py data/yolov3.cfg weights/yolov3.weights weights/yolov3.h5

python tools/convert.py data/yolov3-spp.cfg weights/yolov3-spp.weights weights/yolov3-spp.h5
```

