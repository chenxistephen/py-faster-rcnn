time python ./tools/rpn_im_generate.py --gpu 1 \
  --def models/coco/VGG16/faster_rcnn_end2end/test_rpn.prototxt \
  --net data/faster_rcnn_models/coco_vgg16_faster_rcnn_final.caffemodel \
  --imglist data/flickr30k/imglist.txt \
  --imgdir data/flickr30k/flickr30k_images \
  --out_h5file tmp_rpn_out.h5 \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml
