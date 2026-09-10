# โจทย์ที่ 3: การพยากรณ์รังสีแสงอาทิตย์ด้วย Feature Engineering และ Temporal CNN

## Overview

ได้รับข้อมูลสภาพอากาศและรังสีแสงอาทิตย์รายชั่วโมงของกรุงเทพฯ จาก NASA POWER ให้พยากรณ์ค่า Global Horizontal Irradiance หรือ `ALLSKY_SFC_SW_DWN` ของชั่วโมงถัดไป

โจทย์นี้ต้องสร้าง feature ที่สะท้อนคาบเวลา ตำแหน่งดวงอาทิตย์ สภาพอากาศ ความต่อเนื่องของรังสี และการเปลี่ยนแปลงระยะสั้น ก่อนส่งลำดับย้อนหลัง 48 ชั่วโมงเข้า Temporal CNN ที่สร้างด้วย PyTorch

## Description

ข้อมูลถูกแบ่งตามเวลาเพื่อจำลองการพยากรณ์อนาคตจริง การแบ่งแบบสุ่มหรือ stratified split ไม่เหมาะกับโจทย์นี้ เพราะอาจทำให้ข้อมูลอนาคตรั่วเข้า train การเติม missing value, scaler, lag และ rolling statistics ต้องรักษาทิศทางของเวลา

- `train.csv`: target time ปี 2019-2022 จำนวน 34,895 แถว พร้อม `target`
- `validation.csv`: target time ปี 2023 จำนวน 8,760 แถว พร้อม `target`
- `test.csv`: target time ปี 2024 จำนวน 8,784 แถว ไม่มี `target`

ทุกไฟล์มี timestamp, target time และ feature ชนิดเดียวกัน ชุด test ไม่มีค่าเป้าหมาย แต่สามารถต่อข้อมูลส่วนท้ายของ validation เข้ากับต้น test เพื่อสร้าง lookback 48 ชั่วโมงได้โดยไม่ใช้ข้อมูลอนาคต

กลุ่ม feature ประกอบด้วยค่าจาก NASA POWER, cyclic time features, solar geometry, wind components, clearness index, lag 1/2/3/24/48/168 ชั่วโมง, rolling mean และ standard deviation รวมถึง GHI ramp

## Public Data

โครงสร้างไฟล์ใน `data/public`:

```text
train.csv
validation.csv
test.csv
```

`train.csv` และ `validation.csv` มีคอลัมน์ `target` ส่วน `test.csv` ไม่มีคอลัมน์ดังกล่าว ทั้งสามไฟล์เรียงตามเวลาและต้องไม่ถูกสุ่มสลับก่อนสร้าง sequence

## Evaluation

ใช้ MAE, RMSE และ R-squared โดยรายงานทั้งทุกช่วงเวลาและเฉพาะช่วงที่ค่ารังสีจริงมากกว่า 20

```text
MAE = mean(abs(y_true - y_pred))
RMSE = sqrt(mean((y_true - y_pred)^2))
```

validation ใช้สำหรับเลือก feature, lookback, dilation, dropout, learning rate และ checkpoint ส่วนคำตอบ test ไม่ปรากฏในไฟล์ข้อมูลสาธารณะ

## Submission File

ส่งไฟล์ `submission_solar.csv` จำนวน 8,784 แถว

```csv
id,predicted_ghi
0,0.0
1,0.0
2,5.7
```

- `id`: ลำดับแถวของ `test.csv`
- `predicted_ghi`: ค่า GHI ของชั่วโมงถัดไปที่โมเดลทำนาย และต้องไม่ติดลบ
