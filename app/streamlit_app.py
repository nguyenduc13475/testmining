import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib,numpy as np,pandas as pd,streamlit as st
from src.schema import prepare
st.set_page_config(page_title='HCMUT | Định giá BĐS TP.HCM',page_icon='🏠',layout='wide')
st.title('🏠 Ước lượng giá niêm yết bất động sản TP.HCM')
st.caption('CO3029 – Khai phá dữ liệu | Dự đoán mang tính tham khảo, không phải chứng thư thẩm định giá.')
path=Path('outputs/model.joblib')
if not path.exists():
 st.error('Chưa có mô hình. Chạy: python scripts/download_data.py rồi python scripts/train.py');st.stop()
@st.cache_resource
def load():return joblib.load(path)
artifact=load()
with st.sidebar:
 st.header('Thông tin tài sản')
 typ=st.selectbox('Loại bất động sản',['Nhà','Căn hộ chung cư','Đất','Biệt thự/Nhà liền kề','Shophouse'])
 district=st.text_input('Quận/huyện','Tân Bình')
 ward=st.text_input('Phường/xã','4')
 street=st.text_input('Đường','Cộng Hòa')
 project=st.text_input('Dự án (nếu có)','Không rõ')
 area=st.number_input('Diện tích (m²)',min_value=5.,max_value=5000.,value=80.)
 floors=st.number_input('Số tầng',min_value=0.,max_value=100.,value=2.)
 bedrooms=st.number_input('Số phòng ngủ',min_value=0.,max_value=100.,value=3.)
 bathrooms=st.number_input('Số phòng tắm',min_value=0.,max_value=100.,value=2.)
 frontage=st.number_input('Mặt tiền (m)',min_value=0.,max_value=100.,value=4.)
 depth=st.number_input('Chiều sâu (m)',min_value=0.,max_value=1000.,value=20.)
 road=st.number_input('Đường trước nhà (m)',min_value=0.,max_value=100.,value=5.)
 direction=st.selectbox('Hướng nhà',['Không rõ','Đông','Tây','Nam','Bắc','Đông Bắc','Đông Nam','Tây Bắc','Tây Nam'])
 month=st.number_input('Tháng đăng tin',min_value=1,max_value=12,value=6)
 year=st.number_input('Năm đăng tin',min_value=2025,max_value=2030,value=2026)
 run=st.button('Ước lượng giá',type='primary',use_container_width=True)
if run:
 raw=pd.DataFrame([{'property_type_name':typ,'district_name':district,'ward_name':ward,'street_name':street,'project_name':project,'area':area,'floor_count':floors,'bedroom_count':bedrooms,'bathroom_count':bathrooms,'frontage_width':frontage or np.nan,'house_depth':depth or np.nan,'road_width':road or np.nan,'house_direction':direction,'balcony_direction':'Không rõ','published_at':f'{year}-{month:02d}-15','province_name':'Hồ Chí Minh'}])
 x=prepare(raw,training=False)[artifact['features']]
 value=float(np.exp(np.clip(artifact['model'].predict(x)[0],0,35)))
 q=artifact['log_radius']
 a,b,c=st.columns(3)
 a.metric('Giá dự báo',f'{value/1e9:,.2f} tỷ VND')
 b.metric('Cận dưới khoảng dự báo 90%',f'{value*np.exp(-q)/1e9:,.2f} tỷ')
 c.metric('Cận trên khoảng dự báo 90%',f'{value*np.exp(q)/1e9:,.2f} tỷ')
 st.info('Khoảng dự báo theo split conformal calibration, không phải khoảng tin cậy 95% của trung bình hay cam kết giá giao dịch. Đặc biệt thận trọng khi nhập vị trí hoặc thông số chưa có trong dữ liệu huấn luyện.')
 st.json({'mo_hinh':artifact['winner'],'price_vnd':round(value),'input':raw.iloc[0].to_dict()})
st.divider()
st.subheader('Kết quả đánh giá thực nghiệm (nếu đã huấn luyện)')
if Path('outputs/metrics.csv').exists():
 st.dataframe(pd.read_csv('outputs/metrics.csv'),hide_index=True,use_container_width=True)
 if Path('outputs/actual_vs_pred.png').exists():st.image('outputs/actual_vs_pred.png',caption='Test set – Giá thực tế và giá dự báo')
else:st.write('Chưa có kết quả thực nghiệm.')
