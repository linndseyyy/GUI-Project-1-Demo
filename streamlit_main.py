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
# [Ben] change file name from "lg_prediction.pkl" to "lg_predictor.pkl"
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
    st.subheader("Business Objective")
    st.write("""
    ###### Xây dựng hệ thống/mô hình dự đoán nhằm: 
            1. Phân loại cảm xúc của khách hàng dựa trên các đánh giá (Positive, Neutral, Negative).
            2. Tăng tốc độ và độ chính xác trong việc phản hồi ý kiến của khách hàng.
            3. Hỗ trợ Hasaki.vn và các đối tác cải thiện sản phẩm, dịch vụ, nâng cao sự hài lòng của khách hàng.
    """)  
    st.write("""###### => Yêu cầu: Dùng thuật toán Machine Learning algorithms trong Python để phân loại bình luận tích cực, trung tính và tiêu cực.""")
    st.image("Sentiment Analysis.jpg")

elif choice == 'Build Project':
    st.subheader("Build Project")
    st.write("##### 1. Some data")
    st.dataframe(data[['ma_san_pham','noi_dung_binh_luan', 'label']].head(3))
    st.dataframe(data[['ma_san_pham','noi_dung_binh_luan', 'label']].tail(3))  

    st.write("##### 2. Trực quan hoá Sentiment Analysis")
    st.image("Plot 1.png")
    st.write("""###### => Dữ liệu không cân bằng, cần thực hiện oversample để cân bằng dữ liệu""")
    st.write("""###### Sau khi thực hiện cân bằng dữ liệu""")
    st.image("Plot 2.png")
   
    st.write("##### 3. Build model")
    st.write("""###### Một mô hình hồi quy logistic được xây dựng và đánh giá bằng PySpark để giải quyết bài toán phân loại""")

    st.write("##### 4. Đánh giá")
    st.code("Score train:"+ str(round(score_train,2)) + " vs Score test:" + str(round(score_test,2)))
    st.code("Accuracy:"+str(round(acc,2)))
    st.write("###### Confusion matrix:")
    st.code(cm)
    st.write("###### Classification report:")
    st.code(cr)
    st.code("Roc AUC score:" + str(round(roc,2)))

    # calculate roc curve
    st.write("###### ROC curve")
    fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
    fig, ax = plt.subplots()       
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.plot(fpr, tpr, marker='.')
    st.pyplot(fig)

    st.write("##### 5. Summary: This model is good enough for Ham vs Spam classification.")

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
            y_pred_new = ham_spam_model.predict(x_new)       
            st.code("New predictions (0: Ham, 1: Spam): " + str(y_pred_new))

# [Ben] Added product search functionality
elif choice == 'Product Search':
    st.subheader("Product Search")
    
    product_code = st.text_input("Enter Product Code:")
    
    if st.button("Search"):
        if product_code:            
            product_info = search_product_by_code(product_code)
            print_product_info(product_info)

            if product_info:
                # [Ben] Cool style =)))
                st.title(f"📦 {product_info['ten_san_pham']}")

                # Product info in a nice box
                st.info(f"📦 {product_info['mo_ta']}")
                
                # Rating with stars
                rating = float(product_info['avg_so_sao'])
                st.markdown(f"### ⭐ Rating: {rating:.1f}/5.0")

                # Reviews metrics in expander
                # with st.expander("📊 Review Statistics", expanded=True):
                #     col1, col2 = st.columns(2)
                #     with col1:
                #         st.metric("Positive Reviews", product_info['total_positive'], "👍", delta_color="normal")
                #     with col2:
                #         st.metric("Negative Reviews", product_info['total_negative'], "👎", delta_color="inverse")

                # Create two main columns for the layout
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

                # [Ben] Table style
                # st.write("### Product Details")
    
                # # Create a DataFrame for better table display
                # table_data = {
                #     'Metric': [
                #         'Product Code',
                #         'Product Name',
                #         'Average Rating',
                #         'Positive Reviews',
                #         'Negative Reviews',
                #         'Most Popular Positive Word',
                #         'Most Popular Negative Word'
                #     ],
                #     'Value': [
                #         product_info['ma_san_pham'],
                #         product_info['ten_san_pham'],
                #         product_info['avg_so_sao'],
                #         product_info['total_positive'],
                #         product_info['total_negative'],
                #         product_info['most_popular_positive_word'],
                #         product_info['most_popular_negative_word']
                #     ]
                # }
                
                # df = pd.DataFrame(table_data)
                # st.table(df)
                
                # [Ben] Simpler style
                # st.write("### Product Details")
                # st.write(f"**Product Code:** {product_info['ma_san_pham']}")
                # st.write(f"**Product Name:** {product_info['ten_san_pham']}")
                # st.write(f"**Average Rating:** {product_info['avg_so_sao']}")
                # st.write(f"**Positive Reviews:** {product_info['total_positive']}")
                # st.write(f"**Negative Reviews:** {product_info['total_negative']}")
                # st.write(f"**Most Popular Positive Word:** {product_info['most_popular_positive_word']}")
                # st.write(f"**Most Popular Negative Word:** {product_info['most_popular_negative_word']}")
                
                # Create an attractive pie chart for review distribution
                # fig, ax = plt.subplots()
                # reviews = [int(product_info['total_positive']), int(product_info['total_negative'])]
                # labels = ['Positive', 'Negative']
                # ax.pie(reviews, labels=labels, autopct='%1.1f%%')
                # st.pyplot(fig)
                
            else:
                st.error("Product not found!")
        else:
            st.warning("Please enter a product code")




