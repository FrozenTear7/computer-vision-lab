augment_mode=1

python prepareImages.py coarse $augment_mode
python prepareImages.py fine $augment_mode
python prepareImages.py real $augment_mode