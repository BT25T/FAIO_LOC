# โจทย์ที่ 1: การนับวัตถุจากภาพภายใต้ Domain Shift

## Overview

ฝ่ายตรวจสอบภาพได้รับภาพขาวดำที่ประกอบด้วยสี่เหลี่ยม สามเหลี่ยม และดาว หน้าที่ของคุณคือสร้างโมเดล CNN ด้วย PyTorch เพื่อทำนายจำนวนรูปทรงทั้งหมดในแต่ละภาพ

ข้อมูลฝึกเป็นภาพต้นฉบับที่สะอาด แต่ภาพทดสอบทั้งหมดมาจากกระบวนการสร้างภาพที่ถูกรบกวนด้วย salt-and-pepper noise, Gaussian blur หรือ motion blur ภาพพื้นฐานของชุดทดสอบไม่เคยปรากฏในชุดฝึกหรือชุดตรวจสอบมาก่อน จึงเกิด domain shift ที่ไม่สามารถแก้ได้ด้วยการจำภาพ

สามารถใช้ OpenCV สำหรับตรวจสอบภาพ ทำ preprocessing และสร้าง augmentation ได้ แต่โมเดล CNN และกระบวนการเรียนรู้ต้องสร้างด้วย PyTorch

## Description

ชุดข้อมูลต้นทางมีภาพพื้นฐานสะอาด 1,795 ภาพ แต่ละภาพมีภาพต้นฉบับหนึ่งภาพและภาพรบกวนหลายรูปแบบ การแบ่งข้อมูลทำที่ระดับ `image_id` และใช้ stratification จากช่วงของ `total_shapes` เพื่อรักษาการกระจายจำนวนวัตถุโดยประมาณ พร้อมรับประกันว่า `image_id` ไม่ซ้ำข้ามชุด

- `train.csv` และ `train_images.npz`: ภาพสะอาด 1,255 ภาพ พร้อม `total_shapes`
- `validation.csv` และ `validation_images.npz`: ภาพสะอาด 270 ภาพ พร้อม `total_shapes`
- `test.csv` และ `test_images.npz`: ภาพรบกวน 1,890 ภาพจากภาพพื้นฐานอีก 270 ภาพ โดยไม่มี `total_shapes`

ไฟล์ CSV เก็บ id และ metadata ส่วนอาร์เรย์ภาพเรียงตาม id อยู่ใน key `images` ของไฟล์ NPZ ที่มีชื่อสอดคล้องกัน ชุด train และ validation ไม่มีภาพรบกวนจากแหล่งข้อมูล ผู้ทำโจทย์จึงต้องสร้าง augmentation จากภาพสะอาดเอง หากต้องการให้โมเดลรับมือกับ noise ใน test

ชนิดของภาพรบกวนใน test ได้แก่ `salt_pepper_medium`, `salt_pepper_heavy`, `salt_pepper_extreme`, `gaussian_blur_heavy`, `gaussian_blur_extreme`, `motion_blur` และ `motion_blur_heavy`

## Public Data

โครงสร้างไฟล์ใน `data/public`:

```text
train.csv
train_images.npz
validation.csv
validation_images.npz
test.csv
test_images.npz
```

`train.csv` และ `validation.csv` มีคอลัมน์ `id`, `image_id`, `total_shapes` ส่วน `test.csv` มี `id`, `image_id`, `noise_type`, `bucket_name`, `difficulty` และไม่มี label

## Evaluation

วัดผลด้วย Mean Absolute Error บนคำตอบจำนวนวัตถุจริงของ test

```text
MAE = mean(abs(y_true - y_pred))
```

คะแนนต่ำกว่าแสดงว่าทำนายจำนวนรูปทรงได้แม่นกว่า การแบ่ง validation มีไว้สำหรับเลือก preprocessing, augmentation, architecture และ checkpoint โดยไม่ใช้คำตอบของ test

## Submission File

ส่งไฟล์ `submission_cv.csv` จำนวน 1,890 แถว

```csv
id,predicted_count
0,24
1,24
2,23
```

- `id`: ลำดับภาพใน `test_images.npz`
- `predicted_count`: จำนวนรูปทรงที่โมเดลทำนาย เป็นจำนวนเต็มไม่ติดลบ
