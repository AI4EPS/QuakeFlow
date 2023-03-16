import os

os.system("python ../CCTorch/run.py --data-list1=templates/event_index.txt  --data-path=templates/template.dat --data-format=memmap --config=templates/config.json  --batch-size=1024  --result-path=templates/ccpairs")