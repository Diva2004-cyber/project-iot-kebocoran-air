# Sistem Deteksi Dini Kebocoran Air Rumah Tangga Berbasis IoT dengan Analisis Anomali, Estimasi Kehilangan Air, dan Dashboard Monitoring Real-Time

[cite_start]Proyek ini merupakan sistem *end-to-end* deteksi dini kebocoran air skala rumah tangga yang mengintegrasikan simulasi perangkat keras ESP32, protokol komunikasi data real-time, middleware kecerdasan buatan (*Machine Learning* lokal), serta visualisasi metrik melalui dashboard interaktif Cloud[cite: 91, 93, 95, 96].

[cite_start]Sistem ini didesain menggunakan pendekatan **Scenario B - Cloud Inference**, di mana perangkat *edge* (ESP32) berfokus pada akuisisi data sensor dan kendali aktuator, sementara komputasi model AI yang intensif diproses di sisi server backend Python sebagai middleware[cite: 100, 102, 103, 104].

---

## 📐 Arsitektur Sistem

Alur data dan kontrol sistem berjalan terintegrasi dengan urutan sebagai berikut:
`ESP32 (Wokwi) ➔ MQTT Broker (HiveMQ) ➔ Python Backend (Inference ML) ➔ ThingsBoard Cloud ➔ Dashboard Real-Time`

### Mekanisme Kerja:
1. [cite_start]**ESP32 (Wokwi):** Membaca parameter sensor secara berkala, mengemasnya menjadi payload JSON, dan mempublikasikannya ke broker MQTT[cite: 102, 109, 131]. [cite_start]Perangkat ini juga memonitor status *failsafe* lokal[cite: 102, 191].
2. [cite_start]**MQTT HiveMQ Broker:** Berperan sebagai jembatan sirkulasi data menggunakan topik unik untuk mencegah interferensi data[cite: 101, 119].
3. [cite_start]**Python Backend:** Melakukan *subscribe* telemetry dari ESP32[cite: 104]. [cite_start]Backend kemudian melakukan *data fusion* dengan menggabungkan data sensor real-time dengan baris *UCI Appliances Energy Prediction Dataset* berdasarkan waktu server untuk melengkapi fitur lingkungan[cite: 176]. [cite_start]Setelah itu, backend melakukan inferensi menggunakan dua model *Machine Learning* dan *Rule-Based Safety*[cite: 101, 158].
4. [cite_start]**ThingsBoard Cloud & Aktuasi:** Backend mengirimkan hasil analisis ke platform ThingsBoard Cloud untuk visualisasi dashboard [cite: 101][cite_start], sekaligus mengirimkan perintah balik (*command*) ke topik kontrol ESP32 untuk menggerakkan aktuator[cite: 98, 104].

---

## 🛠️ Spesifikasi Komponen & Hardware (Simulasi Wokwi)

### Pemetaan Pin & Fungsi Komponen
[cite_start]Komponen di dalam simulator Wokwi dikonfigurasi dengan skema pin berikut[cite: 113]:

| Komponen | Pin GPIO | [cite_start]Fungsi Sistem [cite: 108] |
| :--- | :--- | :--- |
| **ESP32 DevKit** | - | [cite_start]Mikrokontroler utama pemroses logika *edge*[cite: 108]. |
| **Potentiometer 1 (pot1)** | `GPIO 34` | [cite_start]Simulasi sensor debit aliran air (*flow sensor*)[cite: 108, 111]. |
| **Potentiometer 2 (pot2)** | `GPIO 35` | [cite_start]Simulasi sensor genangan air (*moisture sensor*)[cite: 108, 111]. |
| **DHT22** | `GPIO 15` | [cite_start]Sensor suhu dan kelembapan udara sekitar[cite: 108, 111]. |
| **Relay** | `GPIO 4` | [cite_start]Aktuator simulasi katup air (*solenoid valve*)[cite: 108, 111]. |
| **Buzzer** | `GPIO 5` | [cite_start]Alarm peringatan darurat lokal[cite: 108, 111]. |
| **LED Hijau** | `GPIO 18` | [cite_start]Indikator kondisi sistem **Normal**[cite: 108, 111]. |
| **LED Kuning** | `GPIO 19` | [cite_start]Indikator kondisi sistem **Waspada**[cite: 108, 111]. |
| **LED Merah** | `GPIO 21` | [cite_start]Indikator kondisi sistem **Bocor / Darurat**[cite: 108, 112]. |

---

## 🧠 Konfigurasi Machine Learning & Sistem Safety

[cite_start]Backend Python tidak bergantung pada API eksternal, melainkan mengeksekusi model *Machine Learning* lokal yang telah dilatih[cite: 96]:

1. [cite_start]**Isolation Forest (`isolation_forest_model.pkl`):** Bertanggung jawab mendeteksi anomali pada pola penggunaan dan debit aliran air[cite: 147, 152, 153].
2. [cite_start]**Random Forest Classifier (`random_forest_model.pkl`):** Mengklasifikasikan status risiko ke dalam 4 target status: `Normal`, `Waspada`, `Bocor`, atau `Darurat`[cite: 147, 154, 161, 162].
3. [cite_start]**StandardScaler (`scaler.pkl`):** Melakukan normalisasi skala fitur sebelum diumpankan ke model klasifikasi[cite: 147, 156, 157].
4. [cite_start]**Rule-Based Safety System:** Sistem proteksi berlapis yang mendominasi (*override*) status menjadi `Darurat` seketika apabila batas kritis fisis terlampaui[cite: 158, 159].

### Aturan Threshold Sinkron (Backend & Wokwi)
[cite_start]Untuk mencegah perbedaan keputusan antara komputasi lokal (*edge*) dan cloud, batas ambang darurat disamakan sebagai berikut[cite: 181, 191]:
- [cite_start]`EMERGENCY_FLOW_THRESHOLD` = **2.5 LPM** [cite: 180, 186]
- [cite_start]`EMERGENCY_MOISTURE_THRESHOLD` = **85.0** [cite: 180, 186]

[cite_start]Jika *flow rate* > 2.5 atau kelembapan genangan >= 85.0, sistem otomatis mengaktifkan penutupan katup (*relay*) dan membunyikan alarm (*buzzer*)[cite: 108, 109, 180].

---

## 📡 Protokol & Payload Komunikasi MQTT

### 1. Broker Kredensial
- [cite_start]**Host:** `broker.hivemq.com` [cite: 117]
- [cite_start]**Port:** `1883` [cite: 117]

### 2. Topik Komunikasi (Strictly Unique)
[cite_start]*Pastikan menggunakan topik di bawah ini pada `Sketch.ino` dan konfigurasi environment untuk menghindari tabrakan data data pada broker publik[cite: 119, 124]:*
- [cite_start]**Telemetry Topic:** `diva237006173/leak/telemetry` [cite: 120]
- [cite_start]**Control Topic:** `diva237006173/leak/control` [cite: 120]

### 3. Skema Payload JSON
- **Telemetry (Dari ESP32 ➔ Backend):**
  ```json
  {
    "flow_rate_lpm": 1.2,
    "moisture_value": 35.0,
    "temperature": 27.0,
    "humidity": 65.0
  }
  
*step-by-step* eksekusi terminal PowerShell yang aman[cite: 108, 110, 131, 134, 149, 221].

Sekarang kamu tinggal memperbarui repositori GitHub-mu dengan teks di atas, bro! Sukses untuk proyek IoT berbasis AI-nya!
