import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from product_search_service import search_product_by_code, print_product_info

# 1. Read data
# [Ben] change to utf-8 for vietnamese text
data = pd.read_csv("final_data3.csv", encoding='utf-8')
# 2. Data pre-processing
source = data['noi_dung_binh_luan']
target = data['label']

# [Ben] Clean NaN values by replacing them with empty string
source = source.fillna('')

text_data = np.array(source)

count = CountVectorizer(max_features=6000)
count.fit(text_data)
bag_of_words = count.transform(text_data)

X = bag_of_words.toarray()

y = np.array(target)

#3. Save models
  
# luu model CountVectorizer (count)
pkl_count = "count_model.pkl"  
with open(pkl_count, 'wb') as file:  
    pickle.dump(count, file)

#4. Load models 
# Đọc model
# import pickle
pkl_filename = "lg_predictor.pkl"  
with open(pkl_filename, 'rb') as file:  
    lg_prediction = pickle.load(file)
# Read count len model
with open(pkl_count, 'rb') as file:  
    count_model = pickle.load(file)

#--------------
# GUI
st.title("Sentiment Analysis Project")
st.write("## Hasaki - Đánh giá tích cực và tiêu cực")

menu = ["Business Objective", "Build Project", "New Prediction", "Product Search"]
choice = st.sidebar.selectbox('Menu', menu)
st.sidebar.write("""#### Thành viên thực hiện:
                 Lê Gia Linh & Phạm Tường Hân""")
st.sidebar.write("""#### Giảng viên hướng dẫn: Khuất Thuỳ Phương""")
st.sidebar.write("""#### Thời gian thực hiện: 12/2024""")
if choice == 'Business Objective':  
    st.subheader("Giới thiệu doanh nghiệp")  
    st.write("""Hasaki.vn - một hệ thống cửa hàng mỹ phẩm chính hãng và dịch vụ chăm sóc sắc đẹp chuyên sâu với mạng lưới rộng khắp Việt Nam. Với một hệ thống website cho phép khách hàng đặt hàng và để lại bình luận, Hasaki.vn có được một cơ sở dữ liệu khách hàng lớn và đầy tiềm năng khai thác.
    """)  
    st.write(""" Giới hạn hiện tại: Dữ liệu đánh giá là văn bản thô và chưa được xử lý một cách tự động ⇒ Yêu cầu quá trình phân tích thủ công, tốn thời gian và dễ xảy ra sai sót.
    """)  
    st.image("Hasaki.jpg")
    st.subheader("Business Objective")
    st.write("""
    ###### Xây dựng hệ thống/mô hình dự đoán nhằm: 
            1. Phân loại cảm xúc của khách hàng dựa trên các đánh giá (Positive, Neutral, Negative).
        2. Tăng tốc độ và độ chính xác trong việc phản hồi ý kiến của khách hàng.
        3. Hỗ trợ Hasaki.vn và các đối tác cải thiện sản phẩm, dịch vụ, nâng cao sự hài lòng của khách hàng.
    """)  
    st.write("""###### Yêu cầu: Dùng thuật toán Machine Learning algorithms trong Python để phân loại bình luận tích cực, trung tính và tiêu cực.""")
    st.image("Sentiment Analysis.jpg")

elif choice == 'Build Project':
    st.subheader("Build Project")
    st.write("##### 1. Một vài dữ liệu")
    st.dataframe(data[['ma_san_pham','noi_dung_binh_luan', 'label']].head(3))
    st.dataframe(data[['ma_san_pham','noi_dung_binh_luan', 'label']].tail(3))  

    st.write("##### 2. Trực quan hoá Sentiment Analysis")
    st.write("###### Wordcloud bình luận")
    st.image("Wordcloud.png")
    st.write("###### Kiểm tra sự cân bằng dữ liệu")
    st.image("Plot 1.png")
    st.write("""###### ⇒ Dữ liệu không cân bằng, cần thực hiện oversample để cân bằng dữ liệu""")
    st.write("""###### Sau khi thực hiện cân bằng dữ liệu""")
    st.image("Plot 2.png")
   
    st.write("##### 3. Xây dựng mô hình")
    st.write("""Xây dựng một mô hình sử dụng đa dạng các thuật toán gồm Naive Bayes, Logistic Regression và Random Forest. Các mô hình được huấn luyện trên các đánh giá của khách hàng về sản phẩm trên website Hasaki.vn để phân loại thành các mức độ cảm xúc.""")

    st.write("##### 4. Đánh giá")
    st.write("""Xây dựng một mô hình sử dụng đa dạng các thuật toán gồm Naive Bayes, Logistic Regression và Random Forest. Các mô hình được huấn luyện trên các đánh giá của khách hàng về sản phẩm trên website Hasaki.vn để phân loại thành các mức độ cảm xúc.""")
    st.write("""###### Độ chính xác và thời gian chạy model""")
    st.image("Model Performance.png") 
    st.write("""###### Confusion Matrix""")
    st.write("""Naive Bayes""")
    st.image("Confusion Matrix for Naive Bayes.png")
    st.write("""Logistic Regression""")
    st.image("Confusion Matrix for Logistic Regression.png")
    st.write("""Random Forest""")
    st.image("Confusion Matrix for Random Forest.png")
    st.write("##### 5.Kết luận: Mô hình Logistic Regression phù hợp nhất đối với Sentiment Analysis của tập dữ liệu của Hasaki.vn.")
    st.write("#### Mô hình Logistic Regression phù hợp nhất đối với Sentiment Analysis của tập dữ liệu của Hasaki.vn.")

elif choice == 'New Prediction':
    st.subheader("Select data")
    flag = False
    lines = None
    type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
    if type=="Upload":
        # Upload file
        uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1, header=None)
            st.dataframe(lines)            
            lines = lines[0]     
            flag = True                          
    if type=="Input":        
        content = st.text_area(label="Input your content:")
        if content!="":
            lines = np.array([content])
            flag = True
    
    if flag:
        st.write("Content:")
        if len(lines)>0:
            st.code(lines)        
            x_new = count_model.transform(lines)        
            y_pred_new = lg_prediction.predict(x_new)       
            st.code("New predictions (0: Negative, 1. Neutral, 2: Positive): " + str(y_pred_new))

# Added product search functionality
elif choice == 'Product Search':
    st.subheader("Product Search")
    
    product_code = st.text_input("Enter Product Code:")
    
    if st.button("Search"):
        if product_code:            
            product_info = search_product_by_code(product_code)
            print_product_info(product_info)

            if product_info:
                st.title(f"📦 {product_info['ten_san_pham']}")

                # Product info in a nice box
                st.info(f"📦 {product_info['mo_ta']}")
                
                # Rating with stars
                rating = float(product_info['avg_so_sao'])
                st.markdown(f"### ⭐ Rating: {rating:.1f}/5.0")

                left_col, right_col = st.columns(2)

                # Left column: Pie Chart
                with right_col:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    reviews = [int(product_info['total_positive']), int(product_info['total_negative'])]
                    labels = ['Positive', 'Negative']
                    colors = ['#2ecc71', '#e74c3c']
                    ax.pie(reviews, labels=labels, autopct='%1.1f%%', colors=colors)
                    plt.title('Review Distribution')
                    st.pyplot(fig)

                # Right column: Review Statistics
                with left_col:
                    with st.expander("📊 Review Statistics", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Positive Reviews", product_info['total_positive'], "👍", delta_color="normal")
                        with col2:
                            st.metric("Negative Reviews", product_info['total_negative'], "👎", delta_color="inverse")


                # Most used words
                with st.expander("📝 Most Used Words", expanded=True):
                    st.metric("Positive Keyword", product_info['most_popular_positive_word'], "✨", delta_color="normal")
                    st.metric("Negative Keyword", product_info['most_popular_negative_word'], "⚠️", delta_color="inverse")
                  
            else:
                st.error("Product not found!")
        else:
            st.warning("Please enter a product code")




