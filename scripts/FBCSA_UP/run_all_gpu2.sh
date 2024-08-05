
# source /share/softwares/anaconda/conda_init.sh
# conda activate dassl_ssdg

bash run_ssdg.sh ssdg_pacs 210 0 Exp4_FBCSA_update_L_thres Exp_configs/Exp4_FBCSA_update_L_thres.yaml &
bash run_ssdg.sh ssdg_digits 300 1 Exp4_FBCSA_update_L_thres Exp_configs/Exp4_FBCSA_update_L_thres.yaml &
bash run_ssdg.sh ssdg_terra 300 2 Exp4_FBCSA_update_L_thres Exp_configs/Exp4_FBCSA_update_L_thres.yaml &
bash run_ssdg.sh ssdg_vlcs 150 3 Exp4_FBCSA_update_L_thres Exp_configs/Exp4_FBCSA_update_L_thres.yaml 


