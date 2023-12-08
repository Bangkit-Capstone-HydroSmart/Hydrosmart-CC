# main.py
import os
import uvicorn
import traceback
import tensorflow as tf
from pydantic import BaseModel
from fastapi import FastAPI, Response, HTTPException, Query
import pandas as pd  # Tambahkan import untuk pandas
from sklearn.preprocessing import StandardScaler


#Load Dataset

app = FastAPI()

# Load the model
model = tf.keras.models.load_model('plant_rec_model.h5')

# Load the DataFrame
excel_path = r'https://storage.googleapis.com/dataset-hydrosmart/DATASET-HIDROPONIK.xlsx'
df_tanaman = pd.read_excel(excel_path, 'TrainTestData')
df_panduan = pd.read_excel(excel_path, 'Tanaman')
scaler = StandardScaler()
df_tanaman_scaled = scaler.fit_transform(df_tanaman[['Luas', 'Suhu', 'PH', 'Kelembapan', 'Penyinaran']])

class RequestText(BaseModel):
    suhu_pengguna: float
    luas_lahan_pengguna: float
    ph_pengguna: int
    kelembapan_pengguna: int
    penyinaran_pengguna: int

@app.get("/")
def index():
    return "Hello world from ML endpoint!"

@app.post("/predict_text")
def predict_text(req: RequestText, response: Response):
    try:
        # In here you will get text sent by the user
        suhu_pengguna = req.suhu_pengguna
        luas_lahan_pengguna = req.luas_lahan_pengguna
        ph_pengguna = req.ph_pengguna
        kelembapan_pengguna = req.kelembapan_pengguna
        penyinaran_pengguna = req.penyinaran_pengguna

        # Prepare your data for the model
        prepared_data = scaler.transform([[luas_lahan_pengguna, suhu_pengguna, ph_pengguna, kelembapan_pengguna, penyinaran_pengguna]])

        # Predict the data
        result_probabilities = model.predict([prepared_data])

        # Ambil kelas dengan probabilitas tertinggi
        num_recommendations = 3
        top_classes_indices = tf.argsort(result_probabilities, axis=1, direction='DESCENDING')[0, :num_recommendations]
       
        # Dapatkan nama tanaman berdasarkan indeks kelas
        recommended_plants = df_tanaman['Nama'].unique()[top_classes_indices]

        return {"api_output": recommended_plants.tolist()}
    
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return {"error": f"Internal Server Error: {str(e)}"}

@app.get("/panduan")
async def get_panduan(
    tanaman_name: str = Query(None, title="Tanaman Name", description="use Capitalize for name.")
):
    try:
        # Check if the tanaman_name parameter is provided
        if not tanaman_name:
            raise HTTPException(status_code=400, detail="Please provide a tanaman_name parameter.")

        # Filter the dataset based on the provided tanaman_name
        result = df_panduan[df_panduan["Tanaman"] == tanaman_name]

        # Check if the result is empty
        if result.empty:
            raise HTTPException(status_code=404, detail=f"No data found for tanaman_name: {tanaman_name}")

        # Extract the required columns from the result
        output_columns = ["Tanaman", "Sistem Budidaya", "Alat & Bahan"]
        result = result[output_columns]

        # Convert the result to a dictionary for JSON serialization
        result_dict = result.to_dict(orient="records")[0]

        return result_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/list_tanaman")
def list_tanaman():
    tanaman_list = df_panduan['Tanaman'].tolist()
    return {"listTanaman": tanaman_list}
        
# Starting the server
if __name__ == "__main__":
    port = os.environ.get("PORT", 8080)
    print(f"Listening to http://0.0.0.0:{port}")
    uvicorn.run(app, host='0.0.0.0', port=port)
