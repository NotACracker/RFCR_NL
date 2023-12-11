python -B main_Scannet_weak_pretrain.py --labeled_point 50 --log_dir rfcrnl-pretrain
python -B main_Scannet_weak_pseudo.py --labeled_point 50 --log_dir rfcrnl-pretrain
python -B main_Scannet_weak_val.py --labeled_point 50 --log_dir rfcnl-pretrain
python -B main_Scannet_weak_train.py --labeled_point 50 --log_dir rfcrnl-iteration1 --load_dir rfcrnl-pretrain --gt_label_path './experiment/Scannetv2/50_points_/rfcrnl-pretrain/gt_50' --pseudo_label_path "./experiment/Scannetv2/50_points_/rfcrnl-pretrain/prediction/pseudo_label"
python -B main_Scannet_weak_pseudo.py --labeled_point 50 --log_dir rfcrnl-iteration1
python -B main_Scannet_weak_val.py --labeled_point 50 --log_dir rfcrnl-iteration1
for i in {2..10}
do
  j=$((i-1))
  python -B main_Scannet_weak_train.py --labeled_point 50 --log_dir rfcrnl-iteration$i --load_dir rfcrnl-iteration${j} --gt_label_path './experiment/Scannetv2/50_points_/rfcrnl-pretrain/gt_50' --pseudo_label_path "./experiment/Scannetv2/50_points_/rfcrnl-iteration${j}/prediction/pseudo_label"
  python -B main_Scannet_weak_pseudo.py --labeled_point 50 --log_dir rfcrnl-iteration$i
  python -B main_Scannet_weak_val.py --labeled_point 50 --log_dir rfcrnl-iteration$i
done