# โจทย์ที่ 2: การจำแนกข่าวด้วย TF-IDF Weighted TextCNN

## Overview

ได้รับข่าวภาษาอังกฤษจาก AG News และต้องจำแนกแต่ละข่าวเป็น 4 หมวด ได้แก่ World, Sports, Business และ Sci/Tech ข้อความดิบมี HTML entities, URL, ตัวเลข, backslash, ช่องว่างไม่สม่ำเสมอ และข่าวซ้ำบางส่วน

ให้ทำความสะอาดข้อความ สร้างคุณลักษณะ TF-IDF และสร้าง TextCNN ด้วย PyTorch โดยสามารถใช้ scikit-learn สำหรับการแบ่งข้อมูล การสร้าง TF-IDF และการวัดผลได้

## Description

ข้อมูลข่าวหลังลบข้อความซ้ำถูกแบ่งจากแหล่งข้อมูลเดียวกันด้วย stratified split ตาม `label` เพื่อให้สัดส่วนทั้ง 4 หมวดใกล้เคียงกันทุกชุด ไฟล์สาธารณะถูกสร้างเป็นไฟล์แยกชัดเจน และ test ไม่มี label

- `train.csv`: ข่าว 95,713 รายการ พร้อม label
- `validation.csv`: ข่าว 12,000 รายการ พร้อม label
- `test.csv`: ข่าว 12,000 รายการ ไม่มี label

คอลัมน์ `label` ใช้ค่า 0 = World, 1 = Sports, 2 = Business และ 3 = Sci/Tech การทำความสะอาด การลบข้อมูลซ้ำ และการแบ่งชุดต้องเกิดก่อน fit vocabulary และ IDF โดย TF-IDF ต้อง fit จากข้อความใน train เท่านั้น

โมเดลรับลำดับ token และค่าน้ำหนัก TF-IDF ของ token ในตำแหน่งเดียวกัน จากนั้นใช้ convolution หลายขนาด kernel เพื่อเรียนรู้รูปแบบ n-gram ก่อนจำแนกหมวดข่าวด้วย PyTorch

## Public Data

โครงสร้างไฟล์ใน `data/public`:

```text
train.csv
validation.csv
test.csv
```

`train.csv` และ `validation.csv` มีคอลัมน์ `id`, `text`, `label` ส่วน `test.csv` มีเพียง `id`, `text` ลำดับแถวของ `test.csv` ตรงกับ id ที่ใช้ใน submission

## Evaluation

ใช้ Accuracy และ Macro-F1 โดย Macro-F1 ให้น้ำหนักแต่ละคลาสเท่ากัน

```text
Macro-F1 = mean(F1_World, F1_Sports, F1_Business, F1_SciTech)
```

ให้ใช้ validation สำหรับเลือกขั้นตอน clean text, vocabulary size, sequence length, kernel sizes, dropout และ checkpoint โดยห้ามใช้คำตอบของ test ระหว่างพัฒนาโมเดล

## Submission File

ส่งไฟล์ `submission_nlp.csv` จำนวน 12,000 แถว

```csv
id,label
17,2
483,0
921,3
```

- `id`: id จาก `test.csv`
- `label`: คลาสที่ทำนาย ต้องเป็น 0, 1, 2 หรือ 3
