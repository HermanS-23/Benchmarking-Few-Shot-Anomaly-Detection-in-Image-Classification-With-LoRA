# Benchmarking-Few-Shot-Anomaly-Detection-in-Image-Classification-With-LoRA
A Thesis Implementation in Partial Fulfilment of the Requirements for the Degree of Bachelor of Engineering in Electronic Engineering The Chinese University of Hong Kong April 2025

# Background
There are 5 python files in this repo

train_CLIP.py is the main file of CLIP model.
train_dinov2.py is the main file of DINOv2 model.
train_CLIP-Lora.py is the main file of CLIP model integrated with LoRA.
train_dinov2-Lora.py is the main file of DINOv2 model integrated with LoRA.
check_labels.py is the execution used to check the details of labeling of the dataset.

Running train_CLIP.py/train_dinov2.py/train_CLIP-Lora.py/train_dinov2-Lora.py will execute 20 runs in total: 5 runs for the 1-shot setting, 5 for the 2-shot setting, 5 for the 4-shot setting, and 5 for the 8-shot setting. The entire process will take approximately 30 to 60 minutes, depending on the number of images in the specified categories. At the end, the results will be saved in an Excel workbook named "data_collected_temp.xlsx" like below.
![image](https://github.com/user-attachments/assets/78c38690-206c-4e43-9721-5b82d4ae467a)
The fields corresponds to below
![image](https://github.com/user-attachments/assets/cf5638c1-4c02-4cbc-9c97-fc9f173d1b28)

# Detail steps to reproduce the results
1. Ensure application environment is Linux. If not, create a virtual environment using wsl or other ways. Of course, needless to say, you should also download all the required libraries, I personally think that using VScode to work on this is easier.

2. Download MVTec-AD dataset (https://www.mvtec.com/company/research/datasets/mvtec-ad) to local.

3. Modify "mvtec_data_path" variable. This variable exists in files of train_CLIP.py, train_dinov2.py, train_CLIP-Lora.py, train_dinov2-Lora.py (Just search the variable name, and you will find it). Then, modify the local directory path for the category of the testing dataset you wish to use.

4. Make sure the function "set_labels(example)" correctly labels the images. Each category of the dataset is further classified; for example, the MVTeC-AD/bottle category includes labels such as "good," "broken_large," "broken_small," and "contamination," indicating the condition of the product in the image. Modify the sentence "if example['labels'] == x:" where x should specify the label number for "good" images.  You can find this out by running check_labels.py and counting the number of images in the stored local directory (pretty primaeval actually, can be smarter).
