# Giới thiệu đồ án
- **Mô tả**: Dự đoán khả năng tốt nghiệp của sinh viên dựa trên các thông tin được cung cấp
- **Input**: dữ liệu trường UIT về thông tin sinh viên, kết quả học tập, kết quả drl, chứng chỉ anh văn, xlhv, …
- **Output**: gồm 2 nhãn
   + Nhãn 0: sinh viên không còn khả năng tốt nghiệp: du học, bỏ học, xử lý học vụ…
   + Nhãn 1: sinh viên có khả năng tốt nghiệp 
# Công cụ sử dụng:
- Đồ án này được viết bằng Python
- Các thư viên cần có: Sklearn, Pandas, Matplotlib, Gdown, Seaborn
# Tập dữ liệu:
![image](https://github.com/Gnartiel/Probability-of-Graduation/assets/105764822/9d943548-a6be-443c-8a27-8ef8b99e44d8)
# Phương pháp thực nghiệm
Đối với tập train và tập test đã chia, nhóm tạo ra thành 10 nhóm khác nhau. Mỗi nhóm sẽ đại diện cho từng giai đoạn học của sinh viên (theo học kỳ): 
- Nhóm 1_sinh viên mới vào trường: gồm có 10 cột: 'namsinh', 'gioitinh', 'noisinh', 'lopsh', 'khoa', 'hedt', 'khoahoc', 'chuyennganh2', 'dien_tt', 'mamh'.
- Nhóm 2_sinh viên có kq hk1: gồm có 10 cột của nhóm 1 và: 'diem_hk1', 'drl_hk1': tổng 12 cột.
- Nhóm 3_sinh viên có kq hk2: gồm có 12 cột của nhóm 2 và: 'diem_hk2', 'drl_hk2’: tổng 14 cột 
- Nhóm 4_sinh viên có kq hk3: gồm có 14 cột của nhóm 3 và:  'diem_hk3', 'drl_hk3': tổng 16 cột
- Nhóm 5_sinh viên có kq hk4: gồm có 16 cột của nhóm 4 và  'diem_hk4', 'drl_hk4': tổng 18 cột
- Nhóm 6_sinh viên có kq hk5: gồm có 18 cột của nhóm 5 và:  'diem_hk5', 'drl_hk5': tổng 20 cột
- Nhóm 7_sinh viên có kq hk6: gồm có 20 cột của nhóm 6 và: 'diem_hk6', 'drl_hk6': tổng 22 cột
- Nhóm 8_sinh viên có kq hk7: gồm có 22 cột của nhóm 7 và: 'diem_hk7', 'drl_hk7': tổng 24 cột
- Nhóm 9_sinh viên có kq hk8: gồm có 24 cột của nhóm 8 và: 'diem_hk8': tổng 25 cột
- Nhóm 10_sinh viên có kq hk8: gồm có 25 cột của nhóm 8 và 1 số thông tin khác:'loaixn', 'tongdiem_av', 'trangthai_av', 'sl_giam', 'xlhv', 'dtb_toankhoa', 'dtb_tichluy', 'sotc_tichluy': tổng 33 cột.
Lần lượt sử dụng để huấn luyện trên các mô hình gồm Logistic Regression, Naive Bayes, Decision Tree, Support Vector Machine, K Nearest Neighbors, Random Forest và kiểm tra đánh giá dựa trên các độ đo Accuracy, Precision, Recall, F1 score.
# Kết quả thực nghiệm
- **Kết quả thực nghiệm trên từng nhóm**:
![image](https://github.com/Gnartiel/Probability-of-Graduation/assets/105764822/fd7eb892-af26-4873-8d3b-e976d0bb5434)![image](https://github.com/Gnartiel/Probability-of-Graduation/assets/105764822/782437d2-b834-4cd1-8c48-fd134a4ad25f)
- **Kết quả thực nghiệm trên từng mô hình**:
![image](https://github.com/Gnartiel/Probability-of-Graduation/assets/105764822/efa8a473-1eca-4596-9f2f-4c412f5c07dc)
![image](https://github.com/Gnartiel/Probability-of-Graduation/assets/105764822/7d637a8f-f352-46fe-bc15-f1ed4106bb58)
# Các sản phầm gồm có:
1. File "Giải trình chỉnh sửa và bổ sung" trình bày những nội dung đã chỉnh sửa và thực hiện thêm theo góp ý của giảng viên --> thư mục Docs
2. Bộ dữ liệu gốc và sau khi xử lý hoàn chỉnh được sử dụng trong đề tài --> thư mục Preprocessed_dataset
   File mô tả chi tiết về dữ liệu -> chứa thông tin data mới thuộc file giải trình, thông tin data cũ thuộc file report ở thư mục Docs
3. Bảng phân công công việc đánh giá mức độ đóng góp của tất cả các thành viên tham gia --> file report thư mục Docs
4. Báo cáo đồ án môn học theo file "Hướng dẫn đồ án cuối kỳ" và góp ý của giảng viên --> thư mục Docs
5. Slides thuyết trình đồ án --> thư mục Slides
6. Chương trình thực nghiệm --> thư mục Code_Demo
7. Các file chứa model và kết quả nằm ở thư mục gốc: file có đuôi .ipynb
