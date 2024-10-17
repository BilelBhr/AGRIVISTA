import React, { useState } from "react";
import axios from "axios";
import logo from './logo.png';
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import "./App.css";

function App() {
  const [geoData, setGeoData] = useState(null);
  const [climateData, setClimateData] = useState(null);
  const [prediction, setPrediction] = useState(null);

  // Fetch geolocation
  const fetchGeolocation = async () => {
    try {
      const response = await axios.get("http://127.0.0.1:8000/geolocation");
      setGeoData(response.data);
    } catch (error) {
      console.error("Error fetching geolocation:", error);
    }
  };

  // Fetch climate data
  const fetchClimateData = async () => {
    if (!geoData) return;

    try {
      const response = await axios.get(
        `http://127.0.0.1:8000/climate-data/?start_year=2017&end_year=2022&latitude=${geoData.latitude}&longitude=${geoData.longitude}`
      );
      setClimateData(response.data.data);
    } catch (error) {
      console.error("Error fetching climate data:", error);
    }
  };

  // Predict using the model
  const fetchPrediction = async () => {
    if (!climateData || !geoData) return;

    const payload = {
      start_year: 2017,
      end_year: 2022,
      latitude: geoData.latitude,
      longitude: geoData.longitude,
    };

    try {
      const response = await axios.post("http://127.0.0.1:8000/predict/", payload);
      setPrediction(response.data);
    } catch (error) {
      console.error("Error fetching prediction:", error);
    }
  };

  return (
    <div className="App">
      <div class="header">
      <a href="#default" className="logo">
        <img src={logo} alt="Company Logo" className="logo-img" />
      </a>
  <div class="header-right">
    <a class="active" href="#home">Home</a>
    <a href="#contact">Contact</a>
    <a href="#about">About</a>
  </div>
</div>
      <h1>Welcome to AGRIVISTA</h1>

      <div className="geolocation-section">
        <button onClick={fetchGeolocation}>Get Geolocation</button>
        <div className="data-display">
          <h2>Geolocation Data</h2>
          {geoData ? (
            <table>
              <thead>
                <tr>
                  <th>Attribute</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Latitude</td>
                  <td>{geoData.latitude}</td>
                </tr>
                <tr>
                  <td>Longitude</td>
                  <td>{geoData.longitude}</td>
                </tr>
              </tbody>
            </table>
          ) : (
            <p>Data will be displayed here after fetching.</p>
          )}
        </div>
      </div>

      <div className="climate-section">
        <button onClick={fetchClimateData} disabled={!geoData}>
          Get Climate Data
        </button>
        <div className="data-display">
          <h2>Climate Data</h2>
          {climateData ? (
            <table>
              <thead>
                <tr>
                  <th>Attribute</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                {climateData.map((data, index) => (
                  <React.Fragment key={index}>
                    <tr>
                      <td>Soil pH</td>
                      <td>{data.Soil_pH}</td>
                    </tr>
                    <tr>
                      <td>Temperature</td>
                      <td>{data.Temperature.toFixed(2)} °C</td>
                    </tr>
                    <tr>
                      <td>Wind Speed</td>
                      <td>{data.Wind_Speed.toFixed(2)} m/s</td>
                    </tr>
                    <tr>
                      <td>Soil Quality</td>
                      <td>{data.Soil_Quality.toFixed(2)}</td>
                    </tr>
                    <tr>
                      <td>Humidity</td>
                      <td>{data.Humidity.toFixed(2)} %</td>
                    </tr>
                    <tr>
                      <td>N</td>
                      <td>{data.N} mg/kg</td>
                    </tr>
                    <tr>
                      <td>P</td>
                      <td>{data.P} mg/kg</td>
                    </tr>
                    <tr>
                      <td>K</td>
                      <td>{data.K} mg/kg</td>
                    </tr>
                    <tr>
                      <td>Soil Type</td>
                      <td>{data.Soil_Type}</td>
                    </tr>
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          ) : (
            <p>Data will be displayed here after fetching.</p>
          )}
        </div>
      </div>

      <div className="prediction-section">
            <button onClick={fetchPrediction} disabled={!climateData}>
                Predict
            </button>
            {prediction && (
                <div className="data-display">
                    <p>Best Crop: {prediction['Best Crop']}</p>
                    <p>Predicted Yield: {prediction['Predicted Yield']}</p>
                </div>
            )}
        </div>
    </div>
  );
}

export default App;
