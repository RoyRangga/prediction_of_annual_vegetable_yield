from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)
app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            province_name=request.form.get('province_name'),
            nama_komoditas=request.form.get('nama_komoditas'),
            Tn=request.form.get('Tn'),
            Tx=request.form.get('Tx'),
            Tavg=request.form.get('Tavg'),
            RH_avg=request.form.get('RH_avg'),
            RR=request.form.get('RR'),
            ss=request.form.get('ss'),
            ff_x=request.form.get('ff_x'),
            ddd_x=request.form.get('ddd_x'),
            ff_avg=request.form.get('ff_avg'),
            luar_wilayah_hektar=request.form.get('luar_wilayah_hektar'),
            tahun=request.form.get('tahun')
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)

        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
    # if you want to call the index page, just type http://127.0.0.1:5000/