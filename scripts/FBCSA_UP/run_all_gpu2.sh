
source /share/softwares/anaconda/conda_init.sh
conda activate dassl_ssdg
bash run_ssdg.sh ssdg_officehome 975 4 update_protos Exp_configs/Exp1_update_prototype.yaml
bash run_ssdg.sh ssdg_pacs 210 4 update_protos Exp_configs/Exp1_update_prototype.yaml