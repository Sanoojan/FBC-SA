
# conda activate dassl_ssdg
# bash run_ssdg.sh ssdg_officehome 975 2 weighted_up_proto
# bash run_ssdg.sh ssdg_pacs 210 2 weighted_up_proto

bash run_ssdg.sh ssdg_vlcs 75 2 baseline Exp_configs/Exp3_FBCSA_check.yaml
bash run_ssdg.sh ssdg_terra 150 2 baseline Exp_configs/Exp3_FBCSA_check.yaml

# bash run_ssdg.sh ssdg_digits 200 2 baseline Exp_configs/Exp3_FBCSA_check.yaml