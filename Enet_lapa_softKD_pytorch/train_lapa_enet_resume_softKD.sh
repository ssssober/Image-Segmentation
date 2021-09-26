CUDA_VISIBLE_DEVICES=0,1 python main_train_softKD.py \
--resume \
--checkpoint_path ./kd/resume_s_model_enet/checkpoint_59_0041000.tar \
--loadteacher \
--ckpt_path_teacher ./kd/t_model_unet/checkpoint_60_0181000.tar \
--mode train \
--in_channels 3 \
--out_channels 11 \
--epochs 100 \
--lrepochs 80,85,90,95:2 \
--dataset lapa \
--w_pi 10.0 \
--model_teacher unet \
--model enet \
--datapath ./LaPa/ \
--trainlist ./src/Lapa_train_new.txt \
--batch_size 20 \
--train_crop_height 320 --train_crop_width 320 \
--logdir ./kd/trained/unet_resume_enet_softkd \
--save_freq 500 \
--summary_freq 20