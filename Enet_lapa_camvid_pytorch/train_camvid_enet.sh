CUDA_VISIBLE_DEVICES=5,6,7 python main_train.py \
--mode train \
--in_channels 3 \
--out_channels 13 \
--epochs 600 \
--lrepochs 300,400,460,500:2 \
--dataset camvid \
--model enet \
--datapath /data/segmentation/camvid-segnet/ \
--trainlist /data/segmentation/camvid-segnet/camSegnet-train-data-651.txt \
--batch_size 30 \
--train_crop_height 360 --train_crop_width 480 \
--logdir ./trained/trained_enet_ori_camseg-new-2 \
--save_freq 1000 \
--summary_freq 20