# main.py
import os
import uvicorn
import traceback
import tensorflow as tf
from pydantic import BaseModel
from fastapi import FastAPI, Response, Body, HTTPException, Query
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# Load the model
model = tf.keras.models.load_model('model90.h5')

# Load the DataFrame
excel_path = r'https://storage.googleapis.com/dataset-hydrosmart/DATASET-HIDROPONIK.xlsx'
df_tanaman = pd.read_excel(excel_path, 'TrainTestData')
df_panduan = pd.read_excel(excel_path, 'Tanaman')
scaler = StandardScaler()
df_tanaman_scaled = scaler.fit_transform(df_tanaman[['Luas', 'Suhu', 'PH', 'Kelembapan', 'Penyinaran']])

class RecomRequest(BaseModel):
    luas_lahan_pengguna: float
    suhu_pengguna: float
    ph_pengguna: float
    kelembapan_pengguna: float
    penyinaran_pengguna: float

class CalcRequest(BaseModel):
    ppm: float
    mass_solute_mg: float
    volume_solution_liters: float

def calculate_ppm(mass_solute_mg, volume_solution_liters):
    return mass_solute_mg / volume_solution_liters

def calculate_mass_solute(ppm, volume_solution_liters):
    return ppm * volume_solution_liters

def calculate_volume_solution(mass_solute_mg, ppm):
    return mass_solute_mg / ppm

@app.get("/")
def index():
    return """
    ML team has created endpoints:
    hydroponic_recommendations: Provides plant recommendations based on user input and the model's predictions, considering the user's environmental conditions.
    hydroponic_calculator: Allows users to calculate PPM (parts per million), mass of solute, or solution volume based on specified parameters.
    CC team has created endpoints:
    list_tanaman: Provides plant list based from dataset.
    panduan: Provides Details of Panduan based from dataset.
    """

@app.post("/hydroponic_recommendations")
def hydroponic_recommendations(req: RecomRequest, response: Response):
    try:

        # Validate input
        if req.luas_lahan_pengguna < 0.05:
            raise HTTPException(status_code=400, detail="Luas lahan harus minimal 0.05")
        if req.suhu_pengguna < 5 or req.suhu_pengguna > 30:
            raise HTTPException(status_code=400, detail="Suhu harus berada dalam rentang 5 - 30")
        if req.ph_pengguna < 0 or req.ph_pengguna > 14:
            raise HTTPException(status_code=400, detail="PH harus berada dalam rentang 0 - 14")
        if req.kelembapan_pengguna < 40 or req.kelembapan_pengguna > 80:
            raise HTTPException(status_code=400, detail="Kelembapan harus berada dalam rentang 40 - 80")
        if req.penyinaran_pengguna < 6 or req.penyinaran_pengguna > 18:
            raise HTTPException(status_code=400, detail="Penyinaran harus berada dalam rentang 6 - 18")

        # # In here you will get text sent by the user
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
        num_recommendations = 10
        top_classes_indices = tf.argsort(result_probabilities, axis=1, direction='DESCENDING')[0, :num_recommendations]
       
        # Dapatkan nama tanaman berdasarkan indeks kelas
        recommended_plants = df_tanaman['Nama'].unique()[top_classes_indices]

        return {"api_output": recommended_plants.tolist()}
    
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return {"error": f"Internal Server Error: {str(e)}"}
    
@app.post("/hydroponic_calculator_ppm")
def hydroponic_calculator_ppm(response: Response, mass_solute_mg: float = Body(...), volume_solution_liters: float = Body(...)):
    try:
        # Handle the calculation for PPM
        result = calculate_ppm(mass_solute_mg, volume_solution_liters)
        return {"result": result}

    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return {"error": f"Internal Server Error: {str(e)}"}

@app.post("/hydroponic_calculator_mass")
def hydroponic_calculator_mass(response: Response, ppm: float = Body(...), volume_solution_liters: float = Body(...)):
    try:
        # Handle the calculation for mass of solute
        result = calculate_mass_solute(ppm, volume_solution_liters)
        return {"result": result}

    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return {"error": f"Internal Server Error: {str(e)}"}

@app.post("/hydroponic_calculator_volume")
def hydroponic_calculator_volume(response: Response, mass_solute_mg: float = Body(...), ppm: float = Body(...)):
    try:
        # Handle the calculation for solution volume
        result = calculate_volume_solution(mass_solute_mg, ppm)
        return {"result": result}

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
    return tanaman_list
        
# Starting the server
if __name__ == "__main__":
    port = os.environ.get("PORT", 8080)
    print(f"Listening to http://0.0.0.0:{port}")
    uvicorn.run(app, host='0.0.0.0', port=port)
