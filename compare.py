import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def decode_multi_class(multi_class_label):
    mask_label = (multi_class_label // 6) % 3
    gender_label = (multi_class_label // 3) % 2
    age_label = multi_class_label % 3
    return mask_label, gender_label, age_label

def decode_pred(mask, gender, age):
    mask_dict = {0: "MASK", 1: "INCORRECT", 2: "NORMAL"}
    gender_dict = {0: "MALE", 1: "FEMALE"}
    age_dict = {0: 'YOUNG', 1: 'MIDDLE', 2: 'OLD'}

    return mask_dict[mask], gender_dict[gender], age_dict[age]

def extract_rows_with_difference(csv_path, column1, column2):
    # CSV 파일을 불러오기
    df = pd.read_csv(csv_path)

    # 두 열이 다른 행을 추출
    different_rows = df[df[column1] != df[column2]]
    print(len(different_rows))
    return different_rows

def load_images_from_folder(folder_paths):
    image_list = []
    for folder_path in folder_paths:
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_path = os.path.join(folder_path, filename)
                image_list.append(image_path)

    return image_list


# 비교할 CSV 파일 경로
csv_path = "/data/ephemeral/maskdata/train/submission.csv"
# 비교할 두 열의 이름
column1 = "ans"
column2 = "result"

# 함수 실행
df = extract_rows_with_difference(csv_path, column1, column2)
print(df,len(df))
folder_path = glob.glob("../maskdata/train/images/*")
image_paths = load_images_from_folder(folder_paths=folder_path)
temp = []

for i in df.id:
    temp.append(i)

error_labels = [0]*18
for i in image_paths:

    if i[:-4] in temp:
        x = df[df['id'] == i[:-4]]
        mask,gender,age = decode_multi_class(int(x.ans))
        error_labels[int(x.ans)] += 1

        # print("predict")
        # print(f"Mask: {x['mask'].values[0]}, Gender: {x['gender'].values[0]}, Age: {x['age'].values[0]}")
        # print("real")
        # m,g,a=(decode_pred(mask,gender,age))
        # print(f"Mask: {m}, Gender: {g}, Age: {a}")
        # print("+++++++++")
        img = mpimg.imread(i)
        # 이미지 시각화
        # plt.imshow(img)
        # plt.axis('off')  # 축 숨기기
        # plt.show()

code_label = []

fig, ax = plt.subplots()
print(error_labels)
for i in range(18):
    mask, gender, age = decode_multi_class(i)
    a, b, c = decode_pred(mask, gender, age)
    code_label.append(f'{a} {b} {c}')

for i, sublist in enumerate(error_labels):
    ax.bar(i, (sublist), label=code_label[i])

ax.set_title('Histogram of Given Data')
ax.set_xlabel('Index')
ax.set_ylabel('Value')
ax.legend()

plt.show()

