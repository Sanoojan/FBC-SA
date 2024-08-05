

# bash run_ssdg.sh ssdg_officehome 1950 6 Exp3_FBCSA_check Exp_configs/Exp3_FBCSA_check.yaml
# bash run_ssdg.sh ssdg_pacs 105 6 Exp3_FBCSA_check Exp_configs/Exp3_FBCSA_check.yaml


bash run_ssdg.sh ssdg_pacs 105 0 Exp4_FBCSA_update_L_thres Exp_configs/Exp4_FBCSA_update_L_thres.yaml &
bash run_ssdg.sh ssdg_digits 150 1 Exp4_FBCSA_update_L_thres Exp_configs/Exp4_FBCSA_update_L_thres.yaml &
bash run_ssdg.sh ssdg_terra 150 2 Exp4_FBCSA_update_L_thres Exp_configs/Exp4_FBCSA_update_L_thres.yaml &
bash run_ssdg.sh ssdg_vlcs 75 3 Exp4_FBCSA_update_L_thres Exp_configs/Exp4_FBCSA_update_L_thres.yaml 

