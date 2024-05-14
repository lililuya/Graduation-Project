python -m onnx.export_onnx \
    --ckpt-path=pretrained/modnet_photographic_portrait_matting.ckpt \
    --output-path=pretrained/modnet_photographic_portrait_matting.onnx


python -m onnx.inference_onnx \
    --image-path=/mnt/sdb/cxh/liwen/EAT_code/MODNet/data/me.jpg \
    --output-path=/mnt/sdb/cxh/liwen/EAT_code/MODNet/data/me_matting.jpg\
    --model-path=/mnt/sdb/cxh/liwen/EAT_code/MODNet/pretrained/modnet.onnx