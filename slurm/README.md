```
mkir relocation
cd relocation
git clone git@github.com:zhuwq0/GrowClust.git
git clone git@github.com:zhuwq0/HypoDD.git
cd ..
```
```
python download_waveform.py
python run_phasenet.py
python run_gamma.py
python convert_hypodd.py && bash run_hypodd_ct.sh
python convert_growclust.py && bash run_growclust_ct.sh
python cut_templates.py && python run_cctorch.py
python convert_hypodd.py --dtcc && bash run_hypodd_cc.sh
python convert_growclust.py --dtcc && bash run_growclust_cc.sh
python run_template_macthing.py
```