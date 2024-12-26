import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency, chi2
import numpy as np

st.set_page_config(layout="wide")

@st.cache_data
def load_data(path):
  dataset = pd.read_csv(path,compression='gzip')
  dataset["datetime"] = pd.to_datetime(dataset["datetime"])
  return dataset.set_index("datetime")

df = load_data("dataset_dashboard.csv.gz")

# tampung semua nama nama variabel yang diperlukan
num_col = df.select_dtypes(include=['number']).columns
categories_order = ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"]
wind_order = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]

# untuk data statistik numerik
data_stat_num = df.describe(include=['number']).T
data_stat_num["var"] = data_stat_num['std']**2
data_stat_num = data_stat_num[["25%","50%","75%","min","mean","max","var","count"]]

# untuk data statistik kategorik
data_stat_cat = df.describe(exclude=['number']).T

# tabel korelasi numerik
korelasi_df = df[num_col].corr()

def stdr_adj(obs,exp):
    P_col = np.sum(exp, axis=0, keepdims=True) / np.sum(exp)  # Proporsi kolom
    P_row = np.sum(exp, axis=1, keepdims=True) / np.sum(exp)  # Proporsi baris
    res = (obs.values - exp) / np.sqrt(exp * (1 - P_col) * (1 - P_row))

    # Normalisasi residual ke skala z-score
    res_zscore = (res - np.mean(res)) / np.std(res)
    res_df = pd.DataFrame(res_zscore, index=obs.index, columns=obs.columns)

    return res_df

option = st.radio("Pilih dataset:",
                  ["Overall", "Aotizhongxin", "Changping", "Dingling","Dongsi",
                    "Guanyuan","Gucheng","Huairou","Nongzhang","Shunyi","Tiantan",
                    "Wanliu","Wanshouxigong"],
                  horizontal=True)

if option != "Overall":
  df = df[df['station'] == option]

@st.fragment
def overview():
  st.title("Dashboard 🚀 Kualitas Udara 🏭 di 12 stasiun beijing (2013 - 2017)")

  st.markdown(""" Dashboard ini adalah hasil dari exploratory data analysis (EDA) mengenai kualitas udara 
   dengan melihat bagaimana kualitas udara dipengaruhi dan bagaimana pola perubahannya di berbagai waktu, 
   analisis dilakukan menggunakan data kualitas udara dari 12 stasiun pemantauan di Beijing. 
   Dataset ini mencakup beberapa hal, mulai dari data konsentrat polutan udara, 
   data meteorologi dan tingkat kualitas udara
   per jam nya. Jangka waktu adalah dari 1 Maret 2013 hingga 28 Februari 2017. 
   Data terdiri 15 kolom dengan 4 kolom data kategorik dan 11 kolom data numerik
   Dalam analisis ini, saya fokus menjawab dua pertanyaan analisis yaitu :
  1. Bagaimana hubungan antar variabel dalam dataset tersebut?
  2. Bagaimana pola temporal data dari waktu ke waktu?
   
   Dashboard yang saya buat bertujuan untuk memberikan gambaran visual dari hasil analisis tersebut,
   sehingga pengguna dapat dengan mudah memahami hubungan antar variabel dan pola temporal yang
   ada dalam dataset.""")

  st.subheader("Informasi Data")
  overview = st.columns([3,7])
  overview[0].markdown("""
    - datetime  : format waktu (yyyy:mm:dd hh)
    - PM2.5     : konsentrat PM2.5 (ug/m^3)
    - PM10      : konsentrat PM10 (ug/m^3)
    - SO2       : konsentrat SO2 (ug/m^3)
    - NO2       : konsentrat NO2 (ug/m^3)
    - CO        : konsentrat CO (ug/m^3)
    - O3        : konsentrat O3 (ug/m^3)
    - TEMP      : Suhu (°C)
    - PRES      : tekanan udara (hPa)
    - DEWP      : temperatur titik embun (°C)
    - RAIN      : presipitasi (mm)
    - wd        : arah mata angin
    - WSPM      : kecepatan angin (m/s)
    - station   : nama stasiun yang diobservasi
    - kategori  : indeks kualitas udara
    - polutan   : polutan dengan tingkat AQI tertinggi
    """)
  with overview[1]:
    rows_per_page = 720
    total_rows = len(df)
    page = st.number_input("Pilih halaman (1 halaman ada 720 baris data/rentang 1 bulan )", 
                            min_value=1, max_value=(total_rows // rows_per_page) + 1, step=1)

    # Menampilkan data per halaman
    start_row = (page - 1) * rows_per_page
    end_row = start_row + rows_per_page
    st.dataframe(df.iloc[start_row:end_row], use_container_width=True)

@st.fragment
def statistik():
  st.subheader("Parameter statistik data")
  var_data = st.selectbox("Pilih variabel",list(filter(lambda x: x != "station", df.columns[::-1])))
  # untuk data tipe kategorik
  if var_data in ["wd","kategori","polutan"]:
    stat_cols = st.columns([2,6])
    indeks_new = ["total","jumlah kategori","top_frekuensi","frekuensi"]

    kategori_df = df[var_data].value_counts().reset_index()
    kategori_df.columns = [var_data, 'jumlah']

    # info nilai parameter statistik
    with stat_cols[0]:
        stat_cols_1 = st.columns(1)
        stat_cols_2 = st.columns(1)
        stat_cols_3 = st.columns(1)
        stat_cols_4 = st.columns(1)
        for i, col in enumerate(stat_cols_1 + stat_cols_2 + stat_cols_3 + stat_cols_4):
          col.metric(label=indeks_new[i], 
                    value=data_stat_cat.loc[var_data, data_stat_cat.columns[i]], 
                    border=True)
    # grafik distribusi data kategorik (pie dan bar chart)
    with stat_cols[1]:
      fig_cat_plot = make_subplots(rows=1, cols=2,
                          specs=[[{'type': 'xy'}, {'type': 'domain'}]])
      # Menambahkan bar chart
      fig_cat_plot.add_trace(go.Bar(x=kategori_df[var_data], 
                          y=kategori_df["jumlah"],
                          name='bar chart',
                          showlegend=False),
                          row=1, col=1)

      fig_cat_plot.add_trace(go.Pie(labels=kategori_df[var_data], 
                      values=kategori_df["jumlah"], 
                      hole=0.4,
                      name="pie chart"),
                      row=1, col=2)

      # Menambahkan layout
      fig_cat_plot.update_layout(title_text=f'Distribusi frekuensi data {var_data}',
                                showlegend=True)
      
      st.plotly_chart(fig_cat_plot,use_container_width=True)
  # untuk data tipe numerik
  else:
    stat_cols = st.columns([3,4])
    format_angka = lambda angka: f"{int(angka)}" if angka.is_integer() else f"{angka:.2f}"
    indeks_new = ["kuartil 1", "median", "kuartil 3","minimum", "rata-rata", "maksimum","variansi","total"]

    # info nilai parameter statistik
    with stat_cols[0]:
      stat_cols_1 = st.columns([1,1,1])
      stat_cols_2 = st.columns([1,1,1])
      stat_cols_3 = st.columns([2,1])
      for i, col in enumerate(stat_cols_1 + stat_cols_2 + stat_cols_3):
        col.metric(label=indeks_new[i], 
                  value=str(format_angka(data_stat_num.loc[var_data, data_stat_num.columns[i]])), 
                  border=True)
    # grafik distribusi data tipe numerik (boxplot dan histogram)
    with stat_cols[1]:
        hist_plot = px.histogram(np.array(df[var_data]), 
                                  nbins=64, 
                                  marginal="box",
                                  title=f"Distribusi data {var_data} sampling rata-rata per-hari")

        hist_plot.layout.height = 400
        st.plotly_chart(hist_plot, use_container_width=True)

@st.fragment
def korelasi():
  st.subheader("korelasi antar variabel numerik")
  heatmap_col, scatter_col = st.columns(2, border=True)
  # grafik untuk heatmap correlation
  with heatmap_col:
    heatmap = px.imshow(korelasi_df,
                  x=korelasi_df.columns,
                  y=korelasi_df.index,
                  title="Heatmap Correlation",
                  color_continuous_scale="viridis",
                  text_auto=".2f")

    heatmap.layout.height = 600
    heatmap.layout.width = 500

    st.plotly_chart(heatmap, use_container_width=True)
  # grafik untuk melihat korelasi yang lebih interaktif
  with scatter_col:
    @st.fragment
    def scatter_corr():
      cols = st.columns(2)
      param1 = cols[0].selectbox('Parameter sumbu X', num_col)
      param2 = cols[1].selectbox('Parameter sumbu Y', num_col)
      
      fig_scatter = px.scatter(df, x=param1, y=param2,
                              color='station',
                              title=f"Hubungan antara {param1} dan {param2}",
                              color_discrete_sequence=px.colors.qualitative.Dark24)

      fig_scatter.layout.height = 500
      st.plotly_chart(fig_scatter, use_container_width=True)

    scatter_corr()
  # deskripsi
  with st.expander("Klik untuk melihat deskripsi gambar"):
    st.markdown("""
    1.  Terdapat hubungan sangat kuat antara partikel halus (PM2.5) dan partikel kasar (PM10), Keduanya cenderung meningkat bersamaan.
    2.  Polutan jenis karbon monoksida (CO) berkorelasi positif kuat dengan PM2.5 dan PM10, menunjukkan bahwa pembakaran tidak sempurna dari bahan bakar fosil adalah sumber utama polutan ini.
    3.  tingkat ozon (O3) berkorelasi rendah negatif dengan jenis polutan lainnya seperti PM2.5, PM10, SO2, NO2, CO.
    4.  ketika tekanan atmosfer (PRES) turun, suhu (TEMP) dan titik embun (DEWP) naik secara bersamaan. Menunjukkan hubungan cuaca pada umumnya
    5.  Kecepatan angin yang tinggi (WSPM) tidak berkaitan langsung dengan peningkatan polutan yang artinya kecepatan angin tidak terlalu mempengaruhi penyebarkan polutan udara.
    6.  curah hujan (RAIN) adalah variabel yang tidak memiliki pengaruh signifikan terhadap semua variabel baik itu variabel meteorologi maupun variabel polutan
    7. faktor meteorologi seperti tekanan armosfer, suhu, titik embun, kecepatan angin bahkan curah hujan tidak secara langsung mempengaruhi peningkatan polusi udara namun berkorelasi dengan polutan jenis ozon (O3)
    """)

  conting_table, stat_visual = st.columns([5,1])
  # grafik untuk tabel kontingensi residual
  with conting_table:
    # membentuk tabel kontingensi
    tabel_kontingensi = pd.crosstab(df["kategori"],df["wd"]).reindex(index=categories_order, columns=wind_order)
    
    result = chi2_contingency(tabel_kontingensi)

    residual_df = stdr_adj(tabel_kontingensi,result[-1])
    # visualisasi dengan heatmap
    heatmap_residual = px.imshow(residual_df,
                                x=tabel_kontingensi.columns,
                                y=tabel_kontingensi.index,
                                title="Hubungan Antara Tingkat Kualitas Udara dengan Arah Mata Angin",
                                text_auto=".2f")

    heatmap_residual.update_layout(xaxis_title = "Arah Mata Angin",
                                  yaxis_title = None,
                                  height=500,
                                  margin=dict(l=0, r=0, t=20, b=0))

    st.plotly_chart(heatmap_residual, use_container_width=True)
  # grafik untuk data data hasil inferensia
  with stat_visual:
    st.metric(label="Chi Square hitung", 
            value=f"{result[0]:.2f}",
            delta=f"\u03C7² = {chi2.ppf(1 - 0.05, result[2]):.2f}",
            border=True)
    
    st.metric(label="P-value", 
            value=f"{result[1]:.2f}",
            delta="-α = 0.05", 
            delta_color="inverse",
            border=True)

    st.metric(label="Derajat kebebasan", 
            value=result[2],
            border=True)
  # deskripsi
  with st.expander("Klik untuk melihat deskripsi grafik"):
    st.write("""
    grafik diatas adalah korelasi antara tingkat kualitas udara dengan arah mata angin
    angka didalam grafik menandakan jika semakin positif maka kejadian kategori "A" dengan
    arah mata angin "B" sering terjadi, dan semakin negatif menunjukkan bahwa kejadian tersebut
    jarang terjadi atau tidak mungkin "A" dipengaruhi oleh "B", jika nilai mendekati 0 maka 
    kejadian tersebut tidak saling berhubungan. Dibawah ini adalah tabel frekuensi atau tabel kontingensi
    hasil observasi dari kedua variabel tersebut.
    """)
    heatmap_kontingensi = px.imshow(tabel_kontingensi,
                            x=tabel_kontingensi.columns,
                            y=tabel_kontingensi.index,
                            text_auto=".2f")
    heatmap_kontingensi.update_layout(xaxis_title = "Arah Mata Angin",
                                      yaxis_title = None,
                                      margin=dict(l=0, r=0, t=20, b=0),
                                      height=600)
    st.plotly_chart(heatmap_kontingensi)

@st.fragment
def temporal():

  @st.fragment
  def time_series_lineplot():
    kolom = st.columns(2)
    parameter1 = kolom[0].multiselect("parameter",num_col,default=["PM2.5","PM10"])
    opsi = kolom[1].selectbox("opsi waktu", ["harian","bulanan","tahunan"])

    if opsi == "harian":
      temporal = df[parameter1].resample("D").median()
    elif opsi == "bulanan":
      temporal = df[parameter1].resample("ME").median()
    elif opsi == "tahunan":
      temporal = df[parameter1].resample("366D").median()

    fig_line = px.line(temporal,
                        color_discrete_sequence=px.colors.qualitative.D3).update_layout(xaxis_title = None,
                                                                                        yaxis_title = f"rata-rata konsentrat {opsi}")
    st.plotly_chart(fig_line)
  
  time_series_lineplot()
  
  month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
  
  # untuk data bulanan
  monthly_data = df[["kategori","polutan"]].copy()
  monthly_data["month"] = monthly_data.index.strftime('%b')

  monthly_polutan = monthly_data.groupby(['month', 'polutan']).size().unstack(fill_value=0)
  monthly_polutan = monthly_polutan.reindex(month_order, axis=0).T/4

  monthly_kategori = monthly_data.groupby(['month', 'kategori']).size().unstack(fill_value=0)
  monthly_kategori = monthly_kategori.reindex(month_order, axis=0).T/4

  # uji chi square kontingensi
  result = chi2_contingency(monthly_kategori)
  # # perhitungan residual standar adjusted
  residuals_df = stdr_adj(monthly_kategori,result[-1])
  
  pie_data = monthly_kategori.reset_index()
  bar_data = monthly_polutan.reset_index()
  bar_data = bar_data[bar_data['polutan'] != 'clean']
  
  cols_temporal = st.columns([1,4,1], border=True)

  @st.fragment
  def time_series_category():
      bulan = st.selectbox('Proporsi AQI dalam bulan', month_order)

      pie_temporal = px.pie(pie_data, values=bulan, names=categories_order,
                            color=categories_order,
                            hole=0.2,
                            color_discrete_sequence=px.colors.qualitative.D3).update_layout(showlegend=False,
                                                                                            height=200,
                                                                                            width=1000,
                                                                                            margin=dict(l=0, r=0, t=0, b=0))
      bar_temporal = px.bar(bar_data, 
                            x='polutan',
                            y=bulan,
                            title=f"frekuensi polutan {bulan}").update_layout(height=200,
                                                                              margin=dict(l=0, r=10, t=50, b=0),
                                                                              xaxis_title = None,
                                                                              yaxis_title = None)

      st.plotly_chart(pie_temporal, use_container_width=False)
      st.plotly_chart(bar_temporal, use_container_width=True)

  with cols_temporal[0]:
    time_series_category()
  with cols_temporal[1]:
    fig_line_bulanan = px.line(residuals_df.T,
                              x=month_order,
                              y=categories_order,
                              markers=True,
                              title="Tingkat Signifikansi Kualitas udara di Tiap Bulan",
                              color_discrete_sequence=px.colors.qualitative.D3)
    fig_line_bulanan.update_layout(shapes=[dict(type="line",
                                                y0=1.96,
                                                y1=1.96,
                                                x0="Jan",
                                                x1="Dec",
                                                line=dict(color="red", dash="dash", width=2)),
                                            dict(type="line",
                                                y0=-1.96,
                                                y1=-1.96,
                                                x0="Jan",
                                                x1="Dec",
                                                line=dict(color="blue", dash="dash", width=2))],
                                    height=300,
                                    xaxis=dict(categoryorder="array",
                                              categoryarray= month_order),
                                    xaxis_title=None,
                                    yaxis_title="nilai residual standar",
                                    margin=dict(l=0, r=0, t=20, b=30))
    
    st.plotly_chart(fig_line_bulanan, use_container_width=True)
    st.code(""" 
        karena α = 0.05 maka batas signifikansi adalah ±1.96 (α/2 = 0.025).
        🟥 menandakan batas atas dengan nilai 1.96,
            jika terdapat nilai yang berada diluar garis (> 1.96) maka dapat disimpulkan 
            kejadian tersebut sering terjadi pada bulan tersebut
        🟦 menandakan batas bawah dengan nilai -1.96, 
            jika terdapat nilai yang berada diluar garis (< -1.96) maka dapat disimpulkan 
            kejadian tersebut jarang terjadi pada bulan tersebut.
        🟥> x <🟦 jika nilai berada diantara merah dan biru maka 
            kejadian tersebut adalah hal biasa (umum)
        """, language='text') 
  with cols_temporal[2]:
    st.metric(label="Chi Square hitung", 
              value=f"{result[0]:.2f}",
              delta=f"\u03C7² = {chi2.ppf(1 - 0.05, result[2]):.2f}",
              border=True)
  
    st.metric(label="P-value", 
            value=f"{result[1]:.2f}",
            delta="-α = 0.05", 
            delta_color="inverse",
            border=True)

    st.metric(label="Derajat kebebasan", 
            value=result[2],
            border=True)


tab1, tab2, tab3 = st.tabs(["Overview Dataset🔎","Correlation Analys📖", "Time Series Analys📈"])

with tab1:
  overview()
  statistik()

with tab2:
  korelasi()

with tab3:
  temporal()