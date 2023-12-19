import os
import glob
import csv

gender_key = {"female" : 1, "male" : 0}

mask_key = {"mask1": 0,
        "mask2": 0,
        "mask3": 0,
        "mask4": 0,
        "mask5": 0,
        "incorrect_mask": 1,
        "normal": 2}

def age_key(n):
  if n <30:
    return 0
  elif n<60:
    return 1
  else:
    return 2

def decode_pred(mask, gender, age):
    mask_dict = {0: "MASK", 1: "INCORRECT", 2: "NORMAL"}
    gender_dict = {0: "MALE", 1: "FEMALE"}
    age_dict = {0: 'YOUNG', 1: 'MIDDLE', 2: 'OLD'}

    return mask_dict[mask], gender_dict[gender], age_dict[age]

def update_csv(path,folder):
    # The data assigned to the list
    id,gender,_,age =folder.split("\\")[-1].split("_")
    
    for mask in os.listdir(folder):
        mask = mask.split(".")[0]
        label = mask_key[mask] * 6 + gender_key[gender] * 3 + age_key(int(age))
        list_data = [folder.split("\\")[-1]+'/'+mask,label]
        
        with open(path, "a", newline="") as f_object:
            writer_object = csv.writer(f_object)
            writer_object.writerow(list_data)
            f_object.close()


# 예시 사용

csv_path = "output.csv"  # 기존 CSV 파일 경로
source_folder = sorted(glob.glob("../maskdata/train/images/*"))
print(source_folder)
for folder in source_folder:
    update_csv(csv_path, folder)

