from __future__ import annotations

import json
import logging
import os
import pickle
import random
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Event, Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt


# ============================================================
# 1. KONFIGURASI LOGGING
# ============================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("LeakDetectionBackend")


# ============================================================
# 2. KONFIGURASI APLIKASI
# ============================================================

@dataclass(frozen=True)
class BackendConfig:
    """
    Konfigurasi utama backend.

    Nilai default dibuat aman untuk pengujian lokal.
    Untuk produksi, gunakan environment variable agar token ThingsBoard
    dan alamat broker tidak ditulis langsung di source code.
    """

    # Path proyek
    base_dir: Path = Path(__file__).resolve().parent
    model_dir_name: str = "model"

    # File model dan schema
    isolation_model_filename: str = "isolation_forest_model.pkl"
    random_forest_model_filename: str = "random_forest_model.pkl"
    scaler_filename: str = "scaler.pkl"
    schema_filename: str = "model_feature_schema.json"
    uci_dataset_filename: str = "energydata_complete.csv"

    # MQTT Hardware Broker
    hardware_broker_host: str = os.getenv("HARDWARE_MQTT_HOST", "broker.hivemq.com")
    hardware_broker_port: int = int(os.getenv("HARDWARE_MQTT_PORT", "1883"))
    hardware_telemetry_topic: str = os.getenv("ESP32_TELEMETRY_TOPIC", "esp32/telemetry")
    hardware_control_topic: str = os.getenv("ESP32_CONTROL_TOPIC", "esp32/control")

    # MQTT ThingsBoard
    thingsboard_host: str = os.getenv("THINGSBOARD_HOST", "mqtt.thingsboard.cloud")
    thingsboard_port: int = int(os.getenv("THINGSBOARD_PORT", "1883"))
    thingsboard_telemetry_topic: str = os.getenv(
        "THINGSBOARD_TELEMETRY_TOPIC",
        "v1/devices/me/telemetry",
    )
    thingsboard_access_token: str = os.getenv("THINGSBOARD_ACCESS_TOKEN", "")

    # Parameter pembacaan sensor
    sensor_interval_seconds: float = float(os.getenv("SENSOR_INTERVAL_SECONDS", "5"))
    active_flow_threshold_lpm: float = float(os.getenv("ACTIVE_FLOW_THRESHOLD_LPM", "0.1"))

    # Rule safety
    emergency_flow_threshold_lpm: float = float(os.getenv("EMERGENCY_FLOW_THRESHOLD_LPM", "2.5"))
    emergency_moisture_threshold: float = float(os.getenv("EMERGENCY_MOISTURE_THRESHOLD", "85.0"))

    # MQTT behavior
    mqtt_keepalive: int = int(os.getenv("MQTT_KEEPALIVE", "60"))
    mqtt_qos: int = int(os.getenv("MQTT_QOS", "1"))

    @property
    def model_dir(self) -> Path:
        return self.base_dir / self.model_dir_name

    @property
    def isolation_model_path(self) -> Path:
        return self.model_dir / self.isolation_model_filename

    @property
    def random_forest_model_path(self) -> Path:
        return self.model_dir / self.random_forest_model_filename

    @property
    def scaler_path(self) -> Path:
        return self.model_dir / self.scaler_filename

    @property
    def schema_path(self) -> Path:
        return self.model_dir / self.schema_filename

    @property
    def uci_dataset_path(self) -> Path:
        return self.base_dir / self.uci_dataset_filename


# ============================================================
# 3. STATE TRACKING BACKEND
# ============================================================

class FlowStateTracker:
    """
    Pelacak stateful untuk durasi aliran air.

    Karena ESP32 mengirim data periodik, backend menghitung durasi
    aliran aktif berdasarkan jumlah paket masuk dan interval sampling.
    """

    def __init__(self, interval_seconds: float, active_threshold_lpm: float) -> None:
        self.interval_seconds = float(interval_seconds)
        self.active_threshold_lpm = float(active_threshold_lpm)
        self._duration_seconds = 0.0
        self._lock = Lock()
        self._last_update_epoch: Optional[float] = None

    def update(self, flow_rate_lpm: float) -> int:
        """
        Memperbarui durasi flow aktif.

        Jika flow_rate_lpm > active_threshold_lpm:
            durasi bertambah sesuai interval pembacaan.
        Jika tidak:
            durasi di-reset ke 0.

        Return:
            duration_flow_min dalam satuan menit integer.
        """
        with self._lock:
            now_epoch = time.time()

            if flow_rate_lpm > self.active_threshold_lpm:
                if self._last_update_epoch is None:
                    elapsed_seconds = self.interval_seconds
                else:
                    elapsed_seconds = max(
                        self.interval_seconds,
                        now_epoch - self._last_update_epoch,
                    )

                self._duration_seconds += elapsed_seconds
            else:
                self._duration_seconds = 0.0

            self._last_update_epoch = now_epoch

            duration_flow_min = int(round(self._duration_seconds / 60.0))
            return max(duration_flow_min, 0)

    def reset(self) -> None:
        with self._lock:
            self._duration_seconds = 0.0
            self._last_update_epoch = None


# ============================================================
# 4. UTILITAS LOAD FILE
# ============================================================

def ensure_file_exists(path: Path, description: str) -> None:
    """
    Memastikan file wajib tersedia sebelum server dijalankan.
    """
    if not path.exists():
        raise FileNotFoundError(f"{description} tidak ditemukan: {path}")


def load_pickle(path: Path, description: str) -> Any:
    """
    Memuat file pickle secara aman.
    """
    ensure_file_exists(path, description)
    with path.open("rb") as file:
        obj = pickle.load(file)
    logger.info("%s berhasil dimuat dari %s", description, path)
    return obj


def load_json(path: Path, description: str) -> Dict[str, Any]:
    """
    Memuat file JSON.
    """
    ensure_file_exists(path, description)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    logger.info("%s berhasil dimuat dari %s", description, path)
    return data


# ============================================================
# 5. PREPROCESSING DATASET UCI
# ============================================================

def load_and_prepare_uci_dataset(path: Path) -> pd.DataFrame:
    """
    Memuat dan membersihkan dataset UCI Appliances Energy Prediction.

    Kolom penting:
    - date
    - Appliances
    - lights
    - T1 sampai T9
    - RH_1 sampai RH_9
    - hour
    - day_of_week
    """
    ensure_file_exists(path, "Dataset UCI energydata_complete.csv")

    df = pd.read_csv(path)

    temperature_cols = [f"T{i}" for i in range(1, 10)]
    humidity_cols = [f"RH_{i}" for i in range(1, 10)]
    required_cols = ["date", "Appliances", "lights"] + temperature_cols + humidity_cols

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Kolom wajib dataset UCI tidak ditemukan: {missing_cols}")

    df = df[required_cols].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    df["hour"] = df["date"].dt.hour.astype(int)
    df["day_of_week"] = df["date"].dt.dayofweek.astype(int)

    numeric_cols = ["Appliances", "lights"] + temperature_cols + humidity_cols + ["hour", "day_of_week"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median(numeric_only=True))

    logger.info("Dataset UCI berhasil dimuat. Jumlah baris: %s, kolom: %s", df.shape[0], df.shape[1])
    return df


# ============================================================
# 6. BACKEND INFERENCE ENGINE
# ============================================================

class LeakDetectionInferenceEngine:
    """
    Mesin inferensi multilapis:
    Layer 1: Isolation Forest
    Layer 2: Random Forest Classifier
    Layer 3: Rule-Based Safety Override
    """

    def __init__(self, config: BackendConfig) -> None:
        self.config = config

        self.isolation_model = load_pickle(
            config.isolation_model_path,
            "Isolation Forest model",
        )
        self.random_forest_model = load_pickle(
            config.random_forest_model_path,
            "Random Forest model",
        )
        self.scaler = load_pickle(
            config.scaler_path,
            "StandardScaler",
        )
        self.schema = load_json(
            config.schema_path,
            "Model feature schema",
        )

        self.isolation_features = self._read_schema_list("isolation_features")
        self.classification_features = self._read_schema_list("classification_features")

        self.uci_df = load_and_prepare_uci_dataset(config.uci_dataset_path)
        self.flow_tracker = FlowStateTracker(
            interval_seconds=config.sensor_interval_seconds,
            active_threshold_lpm=config.active_flow_threshold_lpm,
        )

        self._validate_schema_against_uci()

    def _read_schema_list(self, key: str) -> List[str]:
        """
        Membaca list fitur dari schema JSON.
        """
        value = self.schema.get(key)
        if not isinstance(value, list) or not value:
            raise ValueError(f"Schema JSON tidak memiliki list valid untuk key '{key}'")
        return [str(item) for item in value]

    def _validate_schema_against_uci(self) -> None:
        """
        Validasi awal agar fitur UCI yang dibutuhkan classifier tersedia.
        """
        known_runtime_features = {
            "flow_rate_lpm",
            "moisture_value",
            "duration_flow_min",
            "total_volume_liter",
            "estimated_loss_liter",
            "anomaly_score",
            "is_anomaly",
        }

        missing_in_uci = [
            feature for feature in self.classification_features
            if feature not in self.uci_df.columns and feature not in known_runtime_features
        ]

        missing_iso_known = [
            feature for feature in self.isolation_features
            if feature not in self.uci_df.columns and feature not in known_runtime_features
        ]

        if missing_in_uci:
            raise ValueError(
                "classification_features memuat fitur yang tidak tersedia "
                f"di UCI maupun runtime sensor: {missing_in_uci}"
            )

        if missing_iso_known:
            raise ValueError(
                "isolation_features memuat fitur yang tidak tersedia "
                f"di UCI maupun runtime sensor: {missing_iso_known}"
            )

        logger.info("Validasi schema fitur berhasil.")
        logger.info("Jumlah isolation_features: %s", len(self.isolation_features))
        logger.info("Jumlah classification_features: %s", len(self.classification_features))

    @staticmethod
    def _safe_float(payload: Dict[str, Any], key: str, default: Optional[float] = None) -> float:
        """
        Mengambil nilai float dari payload JSON.
        """
        value = payload.get(key, default)
        if value is None:
            raise ValueError(f"Field '{key}' tidak tersedia pada payload.")
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Field '{key}' harus numerik. Nilai diterima: {value}") from exc

    def get_context_row(self, current_hour: int, day_of_week: int) -> pd.Series:
        """
        Mengambil satu baris UCI yang konteks jamnya sama dengan jam server saat ini.

        Strategi:
        1. Prioritaskan filter hour dan day_of_week.
        2. Jika kosong, fallback ke filter hour saja.
        3. Jika tetap kosong, fallback ke satu baris acak dari seluruh dataset.
        """
        same_hour_day = self.uci_df[
            (self.uci_df["hour"] == current_hour) &
            (self.uci_df["day_of_week"] == day_of_week)
        ]

        if not same_hour_day.empty:
            return same_hour_day.sample(n=1, random_state=random.randint(0, 1_000_000)).iloc[0]

        same_hour = self.uci_df[self.uci_df["hour"] == current_hour]
        if not same_hour.empty:
            return same_hour.sample(n=1, random_state=random.randint(0, 1_000_000)).iloc[0]

        return self.uci_df.sample(n=1, random_state=random.randint(0, 1_000_000)).iloc[0]

    def compute_water_metrics(self, flow_rate_lpm: float, moisture_value: float) -> Dict[str, float]:
        """
        Menghitung fitur fungsional air:
        - duration_flow_min
        - total_volume_liter
        - estimated_loss_liter
        """
        duration_flow_min = self.flow_tracker.update(flow_rate_lpm)
        total_volume_liter = float(flow_rate_lpm * duration_flow_min)

        if flow_rate_lpm > self.config.emergency_flow_threshold_lpm and moisture_value >= self.config.emergency_moisture_threshold:
            estimated_loss_liter = total_volume_liter * 0.90
        elif 0.8 <= flow_rate_lpm <= 2.0 and 50.0 <= moisture_value <= 75.0:
            estimated_loss_liter = total_volume_liter * 0.70
        elif 0.25 <= flow_rate_lpm <= 0.60 and duration_flow_min >= 30 and 30.0 <= moisture_value <= 45.0:
            estimated_loss_liter = max(0.0, flow_rate_lpm * max(duration_flow_min - 10, 0))
        else:
            estimated_loss_liter = 0.0

        return {
            "duration_flow_min": int(duration_flow_min),
            "total_volume_liter": round(float(total_volume_liter), 6),
            "estimated_loss_liter": round(float(estimated_loss_liter), 6),
        }

    def build_fused_record(self, telemetry_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Menggabungkan payload ESP32 dengan konteks UCI berdasarkan waktu aktual server.
        """
        now = datetime.now()
        current_hour = int(now.hour)
        day_of_week = int(now.weekday())

        flow_rate_lpm = self._safe_float(telemetry_payload, "flow_rate_lpm")
        moisture_value = self._safe_float(telemetry_payload, "moisture_value")
        temperature = self._safe_float(telemetry_payload, "temperature", default=np.nan)
        humidity = self._safe_float(telemetry_payload, "humidity", default=np.nan)

        context_row = self.get_context_row(current_hour=current_hour, day_of_week=day_of_week)
        water_metrics = self.compute_water_metrics(flow_rate_lpm, moisture_value)

        fused_record: Dict[str, Any] = {}

        # Salin seluruh fitur konteks dari dataset UCI.
        for column in self.uci_df.columns:
            if column == "date":
                continue
            value = context_row[column]
            if isinstance(value, (np.integer, np.floating)):
                fused_record[column] = float(value)
            else:
                fused_record[column] = value

        # Paksa waktu sesuai waktu server aktual, bukan timestamp historis dataset.
        fused_record["hour"] = current_hour
        fused_record["day_of_week"] = day_of_week

        # Tambahkan pembacaan sensor air aktual dari ESP32.
        fused_record["flow_rate_lpm"] = round(float(flow_rate_lpm), 6)
        fused_record["moisture_value"] = round(float(moisture_value), 6)
        fused_record["duration_flow_min"] = int(water_metrics["duration_flow_min"])
        fused_record["total_volume_liter"] = float(water_metrics["total_volume_liter"])
        fused_record["estimated_loss_liter"] = float(water_metrics["estimated_loss_liter"])

        # Tambahan sensor lingkungan dari ESP32 untuk dikirim ke dashboard.
        # Model tetap menggunakan fitur T1-T9 dan RH_1-RH_9 dari UCI sesuai schema training.
        fused_record["esp32_temperature"] = None if np.isnan(temperature) else round(float(temperature), 6)
        fused_record["esp32_humidity"] = None if np.isnan(humidity) else round(float(humidity), 6)

        # Metadata waktu operasional.
        fused_record["server_timestamp"] = now.isoformat(timespec="seconds")

        return fused_record

    def run_isolation_layer(self, fused_record: Dict[str, Any]) -> Tuple[float, int]:
        """
        Layer 1:
        Menjalankan Isolation Forest untuk mendapatkan anomaly_score dan is_anomaly.
        """
        iso_input = pd.DataFrame([{feature: fused_record[feature] for feature in self.isolation_features}])
        iso_input = iso_input[self.isolation_features]

        anomaly_score = float(self.isolation_model.decision_function(iso_input)[0])
        iso_prediction = int(self.isolation_model.predict(iso_input)[0])
        is_anomaly = 1 if iso_prediction == -1 else 0

        return anomaly_score, is_anomaly

    def run_classification_layer(self, fused_record: Dict[str, Any]) -> str:
        """
        Layer 2:
        Menjalankan Random Forest Classifier setelah StandardScaler.
        """
        classifier_input = pd.DataFrame([
            {feature: fused_record[feature] for feature in self.classification_features}
        ])
        classifier_input = classifier_input[self.classification_features]

        classifier_input_scaled = self.scaler.transform(classifier_input)
        model_status = str(self.random_forest_model.predict(classifier_input_scaled)[0])

        return model_status

    def apply_safety_override(self, fused_record: Dict[str, Any], model_status: str) -> str:
        """
        Layer 3:
        Rule-Based Safety Override.

        Jika flow_rate_lpm > 2.0 dan moisture_value >= 80.0,
        status akhir dipaksa menjadi Darurat.
        """
        flow_rate_lpm = float(fused_record["flow_rate_lpm"])
        moisture_value = float(fused_record["moisture_value"])

        if (
            flow_rate_lpm > self.config.emergency_flow_threshold_lpm
            and moisture_value >= self.config.emergency_moisture_threshold
        ):
            return "Darurat"

        return model_status

    @staticmethod
    def derive_actuator_command(final_status: str) -> Dict[str, Any]:
        """
        Menurunkan perintah aktuator fisik dari final_status.
        """
        valve_status = "CLOSED" if final_status in {"Bocor", "Darurat"} else "OPEN"
        buzzer = 1 if final_status in {"Waspada", "Bocor", "Darurat"} else 0
        relay = 1 if final_status in {"Bocor", "Darurat"} else 0

        return {
            "valve_status": valve_status,
            "buzzer": buzzer,
            "relay": relay,
        }

    def infer(self, telemetry_payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Pipeline inferensi end-to-end.

        Return:
            telemetry_result: payload lengkap untuk ThingsBoard.
            actuator_command: payload ringkas untuk ESP32.
        """
        fused_record = self.build_fused_record(telemetry_payload)

        anomaly_score, is_anomaly = self.run_isolation_layer(fused_record)
        fused_record["anomaly_score"] = round(float(anomaly_score), 8)
        fused_record["is_anomaly"] = int(is_anomaly)

        model_status = self.run_classification_layer(fused_record)
        final_status = self.apply_safety_override(fused_record, model_status)

        actuator_command = self.derive_actuator_command(final_status)

        telemetry_result = dict(fused_record)
        telemetry_result["model_status"] = model_status
        telemetry_result["final_status"] = final_status
        telemetry_result.update(actuator_command)

        return telemetry_result, actuator_command


# ============================================================
# 7. MQTT BACKEND SERVER
# ============================================================

class LeakDetectionBackendServer:
    """
    Server MQTT ganda:
    - Client 1: Hardware broker untuk ESP32.
    - Client 2: ThingsBoard broker untuk dashboard.
    """

    def __init__(self, config: BackendConfig, engine: LeakDetectionInferenceEngine) -> None:
        self.config = config
        self.engine = engine
        self.stop_event = Event()

        self.hardware_client = self._create_mqtt_client(
            client_id=f"leak-backend-hardware-{int(time.time())}",
        )
        self.thingsboard_client = self._create_mqtt_client(
            client_id=f"leak-backend-thingsboard-{int(time.time())}",
        )

        self.hardware_client.on_connect = self.on_hardware_connect
        self.hardware_client.on_disconnect = self.on_hardware_disconnect
        self.hardware_client.on_message = self.on_hardware_message

        self.thingsboard_client.on_connect = self.on_thingsboard_connect
        self.thingsboard_client.on_disconnect = self.on_thingsboard_disconnect

        if not self.config.thingsboard_access_token:
            raise ValueError(
                "Environment variable THINGSBOARD_ACCESS_TOKEN belum diatur. "
                "Isi dengan Access Token device ThingsBoard sebelum menjalankan backend."
            )

        self.thingsboard_client.username_pw_set(
            username=self.config.thingsboard_access_token,
            password=None,
        )

    @staticmethod
    def _create_mqtt_client(client_id: str) -> mqtt.Client:
        """
        Membuat MQTT client yang kompatibel dengan paho-mqtt versi baru.
        """
        try:
            return mqtt.Client(
                client_id=client_id,
                callback_api_version=mqtt.CallbackAPIVersion.VERSION1,
                clean_session=True,
            )
        except (TypeError, AttributeError):
            return mqtt.Client(
                client_id=client_id,
                clean_session=True,
            )

    def connect(self) -> None:
        """
        Menghubungkan kedua MQTT client.
        """
        logger.info(
            "Menghubungkan Hardware MQTT ke %s:%s",
            self.config.hardware_broker_host,
            self.config.hardware_broker_port,
        )
        self.hardware_client.connect(
            self.config.hardware_broker_host,
            self.config.hardware_broker_port,
            keepalive=self.config.mqtt_keepalive,
        )

        logger.info(
            "Menghubungkan ThingsBoard MQTT ke %s:%s",
            self.config.thingsboard_host,
            self.config.thingsboard_port,
        )
        self.thingsboard_client.connect(
            self.config.thingsboard_host,
            self.config.thingsboard_port,
            keepalive=self.config.mqtt_keepalive,
        )

    def start(self) -> None:
        """
        Menjalankan loop MQTT secara non-blocking.
        """
        self.connect()

        self.hardware_client.loop_start()
        self.thingsboard_client.loop_start()

        logger.info("Backend server aktif.")
        logger.info("Subscribe ESP32 telemetry topic: %s", self.config.hardware_telemetry_topic)
        logger.info("Publish ESP32 control topic: %s", self.config.hardware_control_topic)
        logger.info("Publish ThingsBoard topic: %s", self.config.thingsboard_telemetry_topic)

        try:
            while not self.stop_event.is_set():
                time.sleep(0.5)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """
        Menghentikan koneksi MQTT secara rapi.
        """
        logger.info("Mematikan backend server...")

        try:
            self.hardware_client.loop_stop()
            self.thingsboard_client.loop_stop()
            self.hardware_client.disconnect()
            self.thingsboard_client.disconnect()
        except Exception as exc:
            logger.exception("Error saat shutdown MQTT client: %s", exc)

        logger.info("Backend server berhenti.")

    def request_stop(self, signum: Optional[int] = None, frame: Optional[Any] = None) -> None:
        """
        Handler untuk SIGINT/SIGTERM.
        """
        if signum is not None:
            logger.info("Menerima sinyal stop: %s", signum)
        self.stop_event.set()

    # ------------------------------------------------------------
    # Callback Hardware MQTT
    # ------------------------------------------------------------

    def on_hardware_connect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: Dict[str, Any],
        rc: int,
    ) -> None:
        if rc == 0:
            logger.info("Hardware MQTT berhasil terhubung.")
            result, mid = client.subscribe(
                self.config.hardware_telemetry_topic,
                qos=self.config.mqtt_qos,
            )
            logger.info(
                "Subscribe topic '%s' result=%s mid=%s",
                self.config.hardware_telemetry_topic,
                result,
                mid,
            )
        else:
            logger.error("Hardware MQTT gagal terhubung. rc=%s", rc)

    def on_hardware_disconnect(
        self,
        client: mqtt.Client,
        userdata: Any,
        rc: int,
    ) -> None:
        if rc != 0:
            logger.warning("Hardware MQTT terputus tidak normal. rc=%s", rc)
        else:
            logger.info("Hardware MQTT terputus normal.")

    def on_hardware_message(
        self,
        client: mqtt.Client,
        userdata: Any,
        msg: mqtt.MQTTMessage,
    ) -> None:
        """
        Callback utama saat ESP32 mengirim telemetri.
        Seluruh parsing dan inferensi dibungkus try-except agar backend tidak crash.
        """
        raw_payload = msg.payload.decode("utf-8", errors="replace")

        logger.info("Paket masuk dari topic=%s payload=%s", msg.topic, raw_payload)

        try:
            telemetry_payload = json.loads(raw_payload)
            if not isinstance(telemetry_payload, dict):
                raise ValueError("Payload JSON harus berupa object/dictionary.")

            telemetry_result, actuator_command = self.engine.infer(telemetry_payload)

            self.publish_to_thingsboard(telemetry_result)
            self.publish_control_to_esp32(actuator_command)

            logger.info(
                "Inferensi sukses | model_status=%s | final_status=%s | flow=%.3f | moisture=%.3f | anomaly=%s",
                telemetry_result.get("model_status"),
                telemetry_result.get("final_status"),
                float(telemetry_result.get("flow_rate_lpm", 0.0)),
                float(telemetry_result.get("moisture_value", 0.0)),
                telemetry_result.get("is_anomaly"),
            )

        except json.JSONDecodeError as exc:
            logger.error("Payload MQTT bukan JSON valid: %s | raw=%s", exc, raw_payload)

        except KeyError as exc:
            logger.error("Fitur wajib tidak tersedia saat inferensi: %s | raw=%s", exc, raw_payload)

        except ValueError as exc:
            logger.error("Validasi payload/inferensi gagal: %s | raw=%s", exc, raw_payload)

        except Exception as exc:
            logger.exception("Unhandled error saat memproses paket MQTT: %s | raw=%s", exc, raw_payload)

    # ------------------------------------------------------------
    # Callback ThingsBoard MQTT
    # ------------------------------------------------------------

    def on_thingsboard_connect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: Dict[str, Any],
        rc: int,
    ) -> None:
        if rc == 0:
            logger.info("ThingsBoard MQTT berhasil terhubung.")
        else:
            logger.error("ThingsBoard MQTT gagal terhubung. rc=%s", rc)

    def on_thingsboard_disconnect(
        self,
        client: mqtt.Client,
        userdata: Any,
        rc: int,
    ) -> None:
        if rc != 0:
            logger.warning("ThingsBoard MQTT terputus tidak normal. rc=%s", rc)
        else:
            logger.info("ThingsBoard MQTT terputus normal.")

    # ------------------------------------------------------------
    # Publisher
    # ------------------------------------------------------------

    def publish_to_thingsboard(self, telemetry_result: Dict[str, Any]) -> None:
        """
        Mengirim payload lengkap ke ThingsBoard Cloud.
        """
        payload = json.dumps(telemetry_result, ensure_ascii=False)
        info = self.thingsboard_client.publish(
            self.config.thingsboard_telemetry_topic,
            payload=payload,
            qos=self.config.mqtt_qos,
            retain=False,
        )

        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            logger.warning("Publish ke ThingsBoard gagal. rc=%s", info.rc)
        else:
            logger.debug("Publish ke ThingsBoard sukses. mid=%s", info.mid)

    def publish_control_to_esp32(self, actuator_command: Dict[str, Any]) -> None:
        """
        Mengirim payload kontrol ringkas ke ESP32.
        """
        payload = json.dumps(actuator_command, ensure_ascii=False)
        info = self.hardware_client.publish(
            self.config.hardware_control_topic,
            payload=payload,
            qos=self.config.mqtt_qos,
            retain=False,
        )

        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            logger.warning("Publish kontrol ke ESP32 gagal. rc=%s", info.rc)
        else:
            logger.debug("Publish kontrol ke ESP32 sukses. mid=%s", info.mid)


# ============================================================
# 8. SELF-TEST OPSIONAL TANPA MQTT
# ============================================================

def run_local_self_test(engine: LeakDetectionInferenceEngine) -> None:
    """
    Self-test lokal untuk memastikan pipeline model bisa berjalan
    sebelum backend dihubungkan dengan MQTT.
    """
    test_payloads = [
        {
            "flow_rate_lpm": 0.0,
            "moisture_value": 15.0,
            "temperature": 26.5,
            "humidity": 65.0,
        },
        {
            "flow_rate_lpm": 0.4,
            "moisture_value": 35.0,
            "temperature": 26.5,
            "humidity": 65.0,
        },
        {
            "flow_rate_lpm": 1.85,
            "moisture_value": 72.0,
            "temperature": 26.5,
            "humidity": 65.0,
        },
        {
            "flow_rate_lpm": 3.5,
            "moisture_value": 85.0,
            "temperature": 26.5,
            "humidity": 65.0,
        },
    ]

    logger.info("Menjalankan local self-test inferensi...")
    for index, payload in enumerate(test_payloads, start=1):
        telemetry_result, actuator_command = engine.infer(payload)
        logger.info(
            "Self-test #%s | payload=%s | final_status=%s | command=%s",
            index,
            payload,
            telemetry_result.get("final_status"),
            actuator_command,
        )


# ============================================================
# 9. MAIN ENTRY POINT
# ============================================================

def main() -> None:
    """
    Entry point backend server.
    """
    config = BackendConfig()

    logger.info("============================================================")
    logger.info("LEAK DETECTION IOT BACKEND - SCENARIO B CLOUD INFERENCE")
    logger.info("============================================================")
    logger.info("Base directory: %s", config.base_dir)
    logger.info("Model directory: %s", config.model_dir)
    logger.info("Dataset UCI: %s", config.uci_dataset_path)

    try:
        engine = LeakDetectionInferenceEngine(config)

        if os.getenv("RUN_SELF_TEST", "0") == "1":
            run_local_self_test(engine)

        server = LeakDetectionBackendServer(config, engine)

        signal.signal(signal.SIGINT, server.request_stop)
        signal.signal(signal.SIGTERM, server.request_stop)

        server.start()

    except KeyboardInterrupt:
        logger.info("Backend dihentikan oleh pengguna.")

    except Exception as exc:
        logger.exception("Backend gagal dijalankan: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
