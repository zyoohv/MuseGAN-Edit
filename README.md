# MuseGAN-Edit

1. Put your data into dir ./data/lpd5.npz 
2. Run `bash ./scripts/process_data.sh`
2. Run `python3 src/train.py --exp_dir exp/8_beat --params "exp/8_beat/params.yaml" --config "exp/8_beat/config.yaml" --gpu 0,1`
