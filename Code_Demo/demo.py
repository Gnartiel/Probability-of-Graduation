import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import random

random_forest_path = {
    0: './src/random_forest_1.pkl',
    1: './src/random_forest_2.pkl',
    2: './src/random_forest_3.pkl',
    3: './src/random_forest_4.pkl',
    4: './src/random_forest_5.pkl',
    5: './src/random_forest_6.pkl',
    6: './src/random_forest_7.pkl',
    7: './src/random_forest_8.pkl',
    8: './src/random_forest_10.pkl'
}
param_names = ['namsinh', 'gioitinh', 'noisinh', 'lopsh', 'khoa', 'hedt', 'khoahoc', 
        'chuyennganh2', 'dien_tt', 'mamh', 'diem_hk1', 'drl_hk1', 'diem_hk2', 
        'drl_hk2', 'diem_hk3', 'drl_hk3', 'diem_hk4', 'drl_hk4', 'diem_hk5', 
        'drl_hk5', 'diem_hk6', 'drl_hk6', 'diem_hk7', 'drl_hk7', 'diem_hk8', 
        'loaixn', 'tongdiem', 'trangthai', 'sl_giam', 'xlhv', 'dtb_toankhoa', 
        'dtb_tichluy', 'sotc_tichluy']

param_UI_names = {
    'namsinh': "Năm sinh", 
    'gioitinh': "Giới tính", 
    'noisinh': "Nơi sinh", 
    'lopsh': "Lớp sinh hoạt", 
    'khoa': "Khoa", 
    'hedt': "Hệ đào tạo", 
    'khoahoc': "Khóa học", 
    'chuyennganh2': "Chuyên ngành", 
    'dien_tt': "Diện trúng tuyển", 
    'mamh': "Lớp anh văn", 
    'diem_hk1': "Điểm học kì 1", 
    'drl_hk1': "Điểm rèn luyện học kì 1", 
    'diem_hk2': "Điểm học kì 2", 
    'drl_hk2': "Điểm rèn luyện học kì 2", 
    'diem_hk3': "Điểm học kì 3", 
    'drl_hk3': "Điểm rèn luyện học kì 3", 
    'diem_hk4': "Điểm học kì 4", 
    'drl_hk4': "Điểm rèn luyện học kì 4", 
    'diem_hk5': "Điểm học kì 5", 
    'drl_hk5': "Điểm rèn luyện học kì 5", 
    'diem_hk6': "Điểm học kì 6", 
    'drl_hk6': "Điểm rèn luyện học kì 6", 
    'diem_hk7': "Điểm học kì 7", 
    'drl_hk7': "Điểm rèn luyện học kì 7", 
    'diem_hk8': "Điểm học kì 8", 
    'loaixn': "Loại bằng ngoại ngữ", 
    'tongdiem': "Tổng điểm thi anh văn", 
    'trangthai': "Trạng thái anh văn", 
    'sl_giam': "Số lần được miễn giảm học phí", 
    'xlhv': "Số lần bị xử lý học vụ", 
    'dtb_toankhoa': "Điểm trung bình toàn khóa", 
    'dtb_tichluy': "Điểm trung bình tích lũy", 
    'sotc_tichluy': "Số tín chỉ tích lũy"
}

limits = {
    'namsinh':{ "min": 1985, 'max': 2030, 'data_type': 'int' },
    'khoahoc':{ "min": 1, 'max': 20, 'data_type': 'int' },
    'diem_hk1':{ "min": 0.0, 'max': 10.0, 'data_type': 'float' },
    'diem_hk2':{ "min": 0.0, 'max': 10.0, 'data_type': 'float' },
    'diem_hk3':{ "min": 0.0, 'max': 10.0, 'data_type': 'float' },
    'diem_hk4':{ "min": 0.0, 'max': 10.0, 'data_type': 'float' },
    'diem_hk5':{ "min": 0.0, 'max': 10.0, 'data_type': 'float' },
    'diem_hk6':{ "min": 0.0, 'max': 10.0, 'data_type': 'float' },
    'diem_hk7':{ "min": 0.0, 'max': 10.0, 'data_type': 'float' },
    'diem_hk8':{ "min": 0.0, 'max': 10.0, 'data_type': 'float' },
    'drl_hk1':{ "min": 10.0, 'max': 100.0, 'data_type': 'float' },
    'drl_hk2':{ "min": 10.0, 'max': 100.0, 'data_type': 'float' },
    'drl_hk3':{ "min": 10.0, 'max': 100.0, 'data_type': 'float' },
    'drl_hk4':{ "min": 10.0, 'max': 100.0, 'data_type': 'float' },
    'drl_hk5':{ "min": 10.0, 'max': 100.0, 'data_type': 'float' },
    'drl_hk6':{ "min": 10.0, 'max': 100.0, 'data_type': 'float' },
    'drl_hk7':{ "min": 10.0, 'max': 100.0, 'data_type': 'float' },
    'tongdiem':{ "min": 0.0, 'max': 990.0, 'data_type': 'float' },
    'sl_giam':{ "min": 0, 'max': 6, 'data_type': 'int' },
    'xlhv':{ "min": 0, 'max': 6, 'data_type': 'int' },
    'dtb_toankhoa':{ "min": 0.0, 'max': 10.0, 'data_type': 'float' },
    'dtb_tichluy':{ "min": 0.0, 'max': 10.0, 'data_type': 'float' },
    'sotc_tichluy':{ "min": 0, 'max': 500, 'data_type': 'int' },
}

def norm(l, train):
    pd.options.mode.chained_assignment = None 
    # Encode categorical features
    label_encoder = LabelEncoder()
    X_data_list = [train, l]
    for X_data in X_data_list:
        for column in X_data.columns:
            if X_data[column].dtype == 'object' or X_data[column].dtype == 'string':
                X_data[column] = label_encoder.fit_transform(X_data[column])

    # # Standardize features
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    l = scaler.transform(l)

    return l

# Get all user data for training
def get_data(cols):
    data = []
    for col in cols:
        if col == 'gioitinh':
            gender = st.selectbox(param_UI_names[col], ['Nam', 'Nữ'])
            if gender == 'Nam':
                data.append(1)
            elif gender == 'Nữ':
                data.append(0)

        elif col == 'noisinh':
            options = ['TP.HCM', 'Đồng Tháp', 'Hà Tĩnh', 'Quảng Ngãi', 'Khánh Hoà',
                        'Cần Thơ', 'Gia Lai', 'Tiền Giang', 'Sông Bé', 'Kiên Giang',
                        'Lâm Đồng', 'Trà Vinh', 'Quảng Nam', 'Bình Thuận', 'Quảng Bình',
                        'Bến Tre', 'Ninh Thuận', 'Bà Rịa Vũng Tàu', 'Thừa Thiên Huế',
                        'Vĩnh Long', 'Tây Ninh', 'Đồng Nai', 'Thanh Hoá', 'Bình Định',
                        'Hà Nam Ninh', 'Nam Định', 'Đắk Nông', 'An Giang', 'Bình Dương',
                        'Khánh Hòa', 'Long An', 'Thanh Hóa', 'Bình Phước', 'Kon Tum',
                        'Đà Nẵng', 'Ninh Bình', 'Cà Mau', 'Đắk Lắk', 'Quảng Trị',
                        'Hải Hưng', 'Hà Tây', 'Phú Yên', 'Hải Dương', 'Nghệ An',
                        'Thái Bình', 'Minh Hải', 'Bắc Giang', 'Sóc Trăng', 'Hà Nam',
                        'Hưng Yên', 'Campuchia', 'Hải Phòng', 'Bạc Liêu', 'Bắc Ninh',
                        'Hà Nội', 'Vĩnh Phúc', 'Yên Bái', 'Liên Bang Nga', 'Gia lai', 'Tỉnh Nghệ An', 'Lai Châu']
            user_input = st.selectbox(param_UI_names[col], sorted(options))
            data.append(user_input)

        elif col == 'lopsh':
            options = ['KTPM0001', 'HTTT0001', 'KHMT0001', 'MMTT0001', 'MTCL2013', 'CNTT2013', 'KTPM2013', 'KTMT2013', 'KHMT2013', 'ANTT2013',
                        'PMCL2013', 'KTMT0001', 'CNTT0001', 'HTTT2013', 'MMTT2013', 'ANTN2013', 'KHTN2013', 'TMĐT2013', 'CTTT2013', 'KHTN2014',
                        'HTTT2014', 'KTPM2014', 'KTMT2014', 'ANTT2014', 'ANTN2014', 'MTCL2014', 'CNTT2014', 'HTCL2014', 'MMTT2014', 'PMCL2014.1',
                        'KHMT2014', 'KHMT2016.1', 'TMĐT2014', 'CTTT2014', 'PMCL2014.2', 'HTTT2015', 'KHMT2015', 'KTPM2015', 'PMCL2015.1', 'TMĐT2015',
                        'KHTN2015', 'HTCL2015', 'ATTT2015', 'CNTT2015', 'ATTN2015', 'MMTT2015', 'CTTT2015', 'MTCL2015.1', 'PMCL2015.2', 'KTMT2015',
                        'MTCL2015.2', 'PMCL2015.3', 'PMCL2016.1', 'CNTT2016', 'TMĐT2016', 'ATTN2016', 'MTCL2016.1', 'KTMT2016', 'HTCL2016.1', 'MMTT2016',
                        'CTTT2016', 'ATTT2016', 'HTTT2016', 'KHTN2016', 'PMCL2016.2', 'HTCL2016.2', 'KHMT2016.2', 'MTCL2016.2', 'PMCL2016.3', 'KTPM2016',
                        'KTPM2017', 'CNTT2017', 'KHTN2017', 'KHCL2017.1', 'KHMT2017', 'KHCL2017.2', 'ATTN2017', 'KTMT2017', 'HTTT2017', 'HTCL2017.1',
                        'ATTT2017', 'MTCL2017.2', 'PMCL2017.1', 'MTCL2017.1', 'MMCL2017', 'PMCL2017.3', 'ATCL2017', 'PMCL2017.2', 'MMTT2017', 'HTCL2017.2',
                        'CTTT2017', 'PMCL2018.2', 'ATTT2018', 'KHMT2018', 'PMCL2018.1', 'CNTT2018', 'KTPM2018', 'HTTT2018', 'MTCL2018.1', 'CNCL2018.1',
                        'MMTT2018', 'ATCL2018.1', 'KHCL2018.1', 'MMCL2018.1', 'MTCL2018.2', 'HTCL2018.1', 'KTMT2018', 'KHDL2018', 'CTTT2018', 'ATCL2018.2',
                        'KHCL2018.2', 'MMCL2018.2', 'MTCL2018.3', 'KHCL2018.3', 'CNCL2018.2', 'CTTT2019.1', 'ATCL2019.1', 'PMCL2019.1',
                        'CNCL2019.1', 'HTTT2019', 'KTMT2019', 'CNTT2019', 'MMCL2019.1', 'KHCL2019.2', 'MTCL2019.2', 'TMCL2019.1', 'CNCL2019.2', 'TMĐT2019',
                        'MMTT2019', 'KHMT2019', 'KTPM2019', 'CNCL2019.3', 'MMCL2019.2', 'KHCL2019.3', 'ATCL2019.2', 'MTCL2019.3', 'HTCL2019.2', 'CTTT2019.2', 'KHDL2019', 'TMCL2019.2']
            user_input = st.text_input(param_UI_names[col], random.choice(options))
            data.append(user_input)
        
        elif col == 'khoa':
            options = ['CNPM', 'HTTT', 'KHMT', 'MMT&TT', 'KTMT', 'KTTT', 'TMĐT']
            user_input = st.selectbox(param_UI_names[col], sorted(options))
            data.append(user_input)

        elif col == 'hedt':
            options = ['CQUI', 'CTTT', 'CLC', 'KSTN', 'CNTN']
            user_input = st.selectbox(param_UI_names[col], sorted(options))
            data.append(user_input)

        elif col == 'chuyennganh2':
            options = ['D480103', 'D480104', 'D52480104', 'D480101', 'D480102', 'D520214',
                        'D480201', 'D480299', '7480201_CLCN', '7480102', '7480201_KHDL', '7480109']
            user_input = st.selectbox(param_UI_names[col], sorted(options))
            data.append(user_input)

        elif col == 'mamh':
            options = ['AVSC1', 'EN001', 'EN002', 'AVSC', 'ENG02', 'ENG01', 'AVSC2',
                        'ENG04', 'ENG03', 'ENG05']
            user_input = st.selectbox(param_UI_names[col], sorted(options))
            data.append(user_input)

        elif col == 'loaixn':
            options = ['TOEIC', 'IELTS', 'VNU-EPT', 'Cambrigde', 'NHAT', 'TOEFL iBT', 'Không có']
            user_input = st.selectbox(param_UI_names[col], sorted(options))
            if user_input == 'Không có':
                user_input == 'Khong co'
            data.append(user_input)

        elif col == 'dien_tt':
            options = ['B', 'C', 'A']
            user_input = st.selectbox(param_UI_names[col], sorted(options))
            data.append(user_input)

        elif col == 'trangthai':
            user_input = st.selectbox(param_UI_names[col], ['Đã có bằng', 'Chưa có bằng'])
            if user_input == 'Đã có bằng':
                data.append(1)
            elif user_input == 'Chưa có bằng':
                data.append(0)

        else:
            min_value, max_value, dt = limits[col]['min'], limits[col]['max'], limits[col]['data_type']
            user_input = st.number_input(param_UI_names[col], min_value=min_value, max_value=max_value, step= 1 if dt=='int' else 0.01)
            
            input_number = user_input
            if user_input:
                try:
                    input_number = float(user_input)
                except ValueError:
                    input_number = user_input
            user_input = input_number
            data.append(user_input)
    return data

# Collecting all user info
def get_info():
    st.title("Classification with RandomForest")

    hockys = [i for i in range(1,9)]
    hockys.insert(0, "Chưa có học kỳ")
    selected_hocky = st.selectbox("Chọn học kỳ", hockys)
    if selected_hocky == "Chưa có học kỳ":
        st.write(f"Sinh viên chưa hoàn thành học kỳ nào!")
        selected_hocky = 0
    else:
        st.write(f"Sinh viên đã hoàn thành {selected_hocky} học kỳ!")
    st.header("Nhập thông tin cá nhân")

    n_col = len(param_names) if selected_hocky == 8 else 10 + 2 * selected_hocky
    data = get_data(param_names[:n_col])
    return selected_hocky, data

def predict_graduation(selected_hocky, data):
    # load model
    with open(random_forest_path[selected_hocky], 'rb') as file:
        loaded_model = pickle.load(file)
    train = pd.read_excel('./src/datapreprocessing.xlsx')

    n_col = len(param_names) if selected_hocky == 8 else 10 + 2 * selected_hocky
    cols = param_names[:n_col]
    train = train[cols]
    
    df = pd.DataFrame([data], columns=cols)
    normalized_data = norm(df, train)
    
    prediction = loaded_model.predict(normalized_data)
    return prediction[0]

if __name__ == "__main__":
    selected_hocky, input_data = get_info()
    print(input_data)
    if st.button('Dự đoán'):
        result_placeholder = st.empty()
        with st.spinner('Đang tính toán'):
            prediction = predict_graduation(selected_hocky, input_data)
        if prediction == 1:
            result_placeholder.write('Bạn có khả năng tốt nghiệp')
        else:
            result_placeholder.write('Bạn chưa có khả năng tốt nghiệp')





# if selected_hocky == 0:
#     # load model
#     with open('./src/random_forest_1.pkl', 'rb') as file:
#         loaded_model = pickle.load(file)
#     train = pd.read_excel('./src/datapreprocessing.xlsx')
#     kq = []
#     for i in range(10):
#         kq.append(fis(st.text_input(lst[i])))
#     col = ['namsinh', 'gioitinh', 'noisinh', 'lopsh', 'khoa', 'hedt', 'khoahoc', 'chuyennganh2', 'dien_tt', 'mamh']
#     train = train[col]
#     st.write(len(kq))
#     kq = pd.DataFrame([kq],columns = col)
#     kq_df = norm(kq, train)
#     st.write(kq)
#     st.write(loaded_model.predict(kq_df))
#     if loaded_model.predict(kq_df) == [1]:
#         st.write('Bạn có khả năng tốt nghiệp')
#     else:
#         st.write('Bạn chưa có khả năng tốt nghiệp')
# elif selected_hocky == 1:
#     with open('./src/random_forest_2.pkl', 'rb') as file:
#         loaded_model = pickle.load(file)
#     train = pd.read_excel('./src/datapreprocessing.xlsx')
#     kq = []
#     for i in range(12):
#         kq.append(fis(st.text_input(lst[i])))
#     col = [lst[i] for i in range(12)]
#     train = train[col]
#     st.write(len(kq))
#     kq = pd.DataFrame([kq],columns = col)
#     kq_df = norm(kq, train)
#     st.write(kq)
#     st.write(loaded_model.predict(kq_df))
#     if loaded_model.predict(kq_df) == [1]:
#         st.write('Bạn có khả năng tốt nghiệp')
#     else:
#         st.write('Bạn chưa có khả năng tốt nghiệp')
# elif selected_hocky == 2:
#     with open('./src/random_forest_3.pkl', 'rb') as file:
#         loaded_model = pickle.load(file)
#     train = pd.read_excel('./src/datapreprocessing.xlsx')
#     kq = []
#     for i in range(14):
#         kq.append(fis(st.text_input(lst[i])))
#     col = [lst[i] for i in range(14)]
#     train = train[col]
#     st.write(len(kq))
#     kq = pd.DataFrame([kq],columns = col)
#     kq_df = norm(kq, train)
#     st.write(kq)
#     st.write(loaded_model.predict(kq_df))
#     if loaded_model.predict(kq_df) == [1]:
#         st.write('Bạn có khả năng tốt nghiệp')
#     else:
#         st.write('Bạn chưa có khả năng tốt nghiệp')
# elif selected_hocky == 3:
#     with open('./src/random_forest_4.pkl', 'rb') as file:
#         loaded_model = pickle.load(file)
#     train = pd.read_excel('./src/datapreprocessing.xlsx')
#     kq = []
#     for i in range(16):
#         kq.append(fis(st.text_input(lst[i])))
#     col = [lst[i] for i in range(16)]
#     train = train[col]
#     st.write(len(kq))
#     kq = pd.DataFrame([kq],columns = col)
#     kq_df = norm(kq, train)
#     st.write(kq)
#     st.write(loaded_model.predict(kq_df))
#     if loaded_model.predict(kq_df) == [1]:
#         st.write('Bạn có khả năng tốt nghiệp')
#     else:
#         st.write('Bạn chưa có khả năng tốt nghiệp')
# elif selected_hocky == 4:
#     with open('./src/random_forest_5.pkl', 'rb') as file:
#         loaded_model = pickle.load(file)
#     train = pd.read_excel('./src/datapreprocessing.xlsx')
#     kq = []
#     for i in range(18):
#         kq.append(fis(st.text_input(lst[i])))
#     col = [lst[i] for i in range(18)]
#     train = train[col]
#     st.write(len(kq))
#     kq = pd.DataFrame([kq],columns = col)
#     kq_df = norm(kq, train)
#     st.write(kq)
#     st.write(loaded_model.predict(kq_df))
#     if loaded_model.predict(kq_df) == [1]:
#         st.write('Bạn có khả năng tốt nghiệp')
#     else:
#         st.write('Bạn chưa có khả năng tốt nghiệp')
# elif selected_hocky == 5:
#     with open('./src/random_forest_6.pkl', 'rb') as file:
#         loaded_model = pickle.load(file)
#     train = pd.read_excel('./src/datapreprocessing.xlsx')
#     kq = []
#     for i in range(20):
#         kq.append(fis(st.text_input(lst[i])))
#     col = [lst[i] for i in range(20)]
#     train = train[col]
#     st.write(len(kq))
#     kq = pd.DataFrame([kq],columns = col)
#     kq_df = norm(kq, train)
#     st.write(kq)
#     st.write(loaded_model.predict(kq_df))
#     if loaded_model.predict(kq_df) == [1]:
#         st.write('Bạn có khả năng tốt nghiệp')
#     else:
#         st.write('Bạn chưa có khả năng tốt nghiệp')
# elif selected_hocky == 6:
#     with open('./src/random_forest_7.pkl', 'rb') as file:
#         loaded_model = pickle.load(file)
#     train = pd.read_excel('./src/datapreprocessing.xlsx')
#     kq = []
#     for i in range(22):
#         kq.append(fis(st.text_input(lst[i]))) 
#     col = [lst[i] for i in range(22)]
#     train = train[col]
#     st.write(len(kq))
#     kq = pd.DataFrame([kq],columns = col)
#     kq_df = norm(kq, train)
#     st.write(kq)
#     st.write(loaded_model.predict(kq_df))
#     if loaded_model.predict(kq_df) == [1]:
#         st.write('Bạn có khả năng tốt nghiệp')
#     else:
#         st.write('Bạn chưa có khả năng tốt nghiệp')      
# elif selected_hocky == 7:
#     with open('./src/random_forest_8.pkl', 'rb') as file:
#         loaded_model = pickle.load(file)
#     train = pd.read_excel('./src/datapreprocessing.xlsx')
#     kq = []
#     for i in range(24):
#         kq.append(fis(st.text_input(lst[i])))  
#     col = [lst[i] for i in range(24)]
#     train = train[col]
#     st.write(len(kq))
#     kq = pd.DataFrame([kq],columns = col)
#     kq_df = norm(kq, train)
#     st.write(kq)
#     st.write(loaded_model.predict(kq_df))
#     if loaded_model.predict(kq_df) == [1]:
#         st.write('Bạn có khả năng tốt nghiệp')
#     else:
#         st.write('Bạn chưa có khả năng tốt nghiệp')      
# elif selected_hocky == 8:
#     with open('./src/random_forest_10.pkl', 'rb') as file:
#         loaded_model = pickle.load(file)
#     train = pd.read_excel('./src/datapreprocessing.xlsx')
#     kq = []
#     for i in range(len(lst)):
#         kq.append(fis(st.text_input(lst[i]))) 
#     col = [lst[i] for i in range(len(lst))]
#     train = train[col]
#     st.write(len(kq))
#     kq = pd.DataFrame([kq],columns = col)
#     kq_df = norm(kq, train)
#     st.write(kq)
#     st.write(loaded_model.predict(kq_df))
#     if loaded_model.predict(kq_df) == [1]:
#         st.write('Bạn có khả năng tốt nghiệp')
#     else:
#         st.write('Bạn chưa có khả năng tốt nghiệp')