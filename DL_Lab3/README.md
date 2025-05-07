# NYCU Computer Vision 2025 Spring Lab 3
- StudentID: 313554002
- Name: 陳子揚

# Setup and run
1. Clone this folder and set the path into it.
2. Put data into the folder.
```
 DL_Lab2 (You should be here)
   ├── data (add data here)
   |  ├── test_realease
   |  ├── train
   |  └── test_image_name_to_ids.json
   ├── detection
   |  ├── coco_eval.py
    ...
   |  ├── transforms.py
   |  └── utils.py
   ├── dataset.py
   ├── enviroment.yaml
   ├── hw3.ipynb
   ├── infrence.py
   ├── model.py
   └── RLE.py
```
3. Open anaconda prompt at that path. 
4. Enter the following command at anaconda prompt
```
conda env create -f environment.yml -n my_DL_Lab3
```
5. Open "hw2.ipynb" in vscode
6. Change the respective file path.
7. Run first 4 cell with environment in "my_DL_Lab3".
8. Change the file name of pth name(model auto saved in "output/trial_(num)").
9. Run the remainder cell.
