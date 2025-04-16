# NYCU Computer Vision 2025 Spring Lab 2
- StudentID: 313554002
- Name: 陳子揚

# Setup and run
1. Clone this folder and set the path into it.
2. Put data into the folder.
```
 DL_Lab2 (You should be here)
   ├── data (add data here)
   |  ├── test
   |  ├── train
   |  ├── val
   |  └── nothing.txt
   ├── dataset.py
   ├── enviroment.yaml
   ├── hw2.ipynb
   ├── infrence.py
   └── model.py
```
3. Open anaconda prompt at that path. 
4. Enter the following command at anaconda prompt
```
conda env create -f environment.yml -n my_DL_Lab2
```
5. Open "hw2.ipynb" in vscode
6. Change the respective file path.
7. Run all cell expect last cell with environment in "my_DL_Lab2".
8. Change the file name of pth name(model auto saved in "output/trial_(num)").
9. Run the last cell.
