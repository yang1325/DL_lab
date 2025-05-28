# NYCU Computer Vision 2025 Spring Lab 4
- StudentID: 313554002
- Name: 陳子揚

# Setup and run
1. Clone this folder and set the path into it.
2. Put data into the folder.
```
 DL_Lab4 (You should be here)
   ├── __pychache__
   ├── ckpt
   ├── data
   |  ├── hw4_realse_dataset (add data at here)
   |  └── nothing.txt.txt
   ├── data_dir
   ├── net
   ├── output
   ├── utils
   ├── INSTALL.md
   ├── LICENSE.md
   ├── ...
   └── RLE.py
```
3. Open anaconda prompt at that path. 
4. Enter the following command at anaconda prompt
```
conda env create -f env.yml
```
5. Run the following  command at anaconda prompt
```
conda activate promptir
python train.py
python infer.py #(change the model path by argument --model if you want)
```
