"""
Locust Load Testing Script for EcoSight Wildlife Monitoring API
Author: EcoSight Team
Date: 2025-11-17

This script simulates a flood of requests to test API performance under load.
It measures latency, response time, and throughput with different numbers of users.

Usage:
    locust -f locustfile.py --host=http://localhost:8000
    
    Then open http://localhost:8089 in your browser to configure and start the test.

Features:
- Simulates realistic user behavior
- Tests prediction endpoint with audio files
- Tests status and metrics endpoints
- Measures response times and error rates
- Supports scaling to thousands of concurrent users
"""

from locust import HttpUser, task, between, events
import random
import json
import time
import os
from pathlib import Path
import io
import wave
import numpy as np

# Global metrics storage
response_times = []
request_count = 0
error_count = 0


class WildlifeMonitoringUser(HttpUser):
    """
    Simulates a user interacting with the Wildlife Monitoring API
    """
    
    # Wait time between tasks (1-3 seconds)
    wait_time = between(1, 3)
    
    def on_start(self):
        """Called when a user starts"""
        self.user_id = random.randint(1000, 9999)
        print(f"User {self.user_id} started")
    
    @task(5)
    def get_status(self):
        """
        Test the status endpoint
        Weight: 5 (higher frequency)
        """
        with self.client.get("/status", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status check failed: {response.status_code}")
    
    @task(3)
    def get_metrics(self):
        """
        Test the metrics endpoint
        Weight: 3 (medium frequency)
        """
        with self.client.get("/metrics", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Metrics fetch failed: {response.status_code}")
    
    @task(10)
    def predict_audio(self):
        """
        Test the prediction endpoint with synthetic audio
        Weight: 10 (highest frequency - main endpoint)
        """
        # Generate synthetic audio file
        audio_data = self.generate_synthetic_audio()
        
        files = {
            'file': ('test_audio.wav', audio_data, 'audio/wav')
        }
        
        start_time = time.time()
        
        with self.client.post("/predict", files=files, catch_response=True) as response:
            response_time = (time.time() - start_time) * 1000  # ms
            
            global response_times, request_count
            response_times.append(response_time)
            request_count += 1
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    if result.get('success'):
                        response.success()
                    else:
                        response.failure("Prediction not successful")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                global error_count
                error_count += 1
                response.failure(f"Prediction failed: {response.status_code}")
    
    @task(1)
    def health_check(self):
        """
        Test the health check endpoint
        Weight: 1 (low frequency)
        """
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    def generate_synthetic_audio(self, duration=2, sample_rate=16000):
        """
        Generate synthetic audio data for testing
        
        Args:
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
        
        Returns:
            BytesIO object containing WAV audio data
        """
        # Generate random noise (simulating audio)
        samples = np.random.uniform(-0.5, 0.5, int(duration * sample_rate))
        samples = samples.astype(np.float32)
        
        # Convert to 16-bit PCM
        audio_int16 = (samples * 32767).astype(np.int16)
        
        # Create WAV file in memory
        audio_bytes = io.BytesIO()
        with wave.open(audio_bytes, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        audio_bytes.seek(0)
        return audio_bytes


class PowerUser(HttpUser):
    """
    Simulates a power user making more frequent requests
    """
    
    wait_time = between(0.5, 1.5)  # Faster requests
    
    @task
    def rapid_predictions(self):
        """Make rapid prediction requests"""
        audio_data = self.generate_synthetic_audio()
        files = {'file': ('test_audio.wav', audio_data, 'audio/wav')}
        
        with self.client.post("/predict", files=files, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed: {response.status_code}")
    
    def generate_synthetic_audio(self, duration=1, sample_rate=16000):
        """Generate short synthetic audio"""
        samples = np.random.uniform(-0.5, 0.5, int(duration * sample_rate))
        samples = samples.astype(np.float32)
        audio_int16 = (samples * 32767).astype(np.int16)
        
        audio_bytes = io.BytesIO()
        with wave.open(audio_bytes, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        audio_bytes.seek(0)
        return audio_bytes


# ============================================================================
# EVENT HOOKS FOR CUSTOM METRICS
# ============================================================================

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts"""
    print("\n" + "="*70)
    print("🚀 LOAD TEST STARTING")
    print("="*70)
    print(f"Target host: {environment.host}")
    print(f"User classes: WildlifeMonitoringUser, PowerUser")
    print("="*70 + "\n")
    
    global response_times, request_count, error_count
    response_times = []
    request_count = 0
    error_count = 0


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops - print summary statistics"""
    print("\n" + "="*70)
    print("📊 LOAD TEST SUMMARY")
    print("="*70)
    
    if response_times:
        print(f"Total Requests: {request_count}")
        print(f"Total Errors: {error_count}")
        print(f"Error Rate: {(error_count/request_count)*100:.2f}%")
        print(f"\nResponse Times (ms):")
        print(f"  Min:     {min(response_times):.2f}")
        print(f"  Max:     {max(response_times):.2f}")
        print(f"  Mean:    {np.mean(response_times):.2f}")
        print(f"  Median:  {np.median(response_times):.2f}")
        print(f"  P95:     {np.percentile(response_times, 95):.2f}")
        print(f"  P99:     {np.percentile(response_times, 99):.2f}")
    else:
        print("No response time data collected")
    
    print("="*70 + "\n")


# ============================================================================
# CUSTOM TEST SCENARIOS
# ============================================================================

class StressTestUser(HttpUser):
    """
    Stress test scenario - maximum load
    """
    
    wait_time = between(0.1, 0.5)  # Very fast requests
    
    @task
    def stress_predict(self):
        """Stress test the prediction endpoint"""
        audio_data = io.BytesIO(b'\x00' * 1000)  # Minimal data
        files = {'file': ('test.wav', audio_data, 'audio/wav')}
        
        self.client.post("/predict", files=files)


# ============================================================================
# EXAMPLE TEST CONFIGURATIONS
# ============================================================================

"""
Example test configurations:

1. Light Load (5-10 users):
   locust -f locustfile.py --host=http://localhost:8000 --users=10 --spawn-rate=2 --run-time=2m

2. Medium Load (50-100 users):
   locust -f locustfile.py --host=http://localhost:8000 --users=100 --spawn-rate=10 --run-time=5m

3. Heavy Load (500+ users):
   locust -f locustfile.py --host=http://localhost:8000 --users=500 --spawn-rate=50 --run-time=10m

4. Stress Test (1000+ users):
   locust -f locustfile.py --host=http://localhost:8000 --users=1000 --spawn-rate=100 --run-time=15m

5. Web UI Mode (interactive):
   locust -f locustfile.py --host=http://localhost:8000
   Then open http://localhost:8089

6. Headless Mode with CSV output:
   locust -f locustfile.py --host=http://localhost:8000 --users=100 --spawn-rate=10 --run-time=5m --headless --csv=results

7. Docker Container Testing:
   locust -f locustfile.py --host=http://localhost:80 --users=100 --spawn-rate=10 --run-time=5m
"""

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 LOCUST LOAD TESTING FOR ECOSIGHT WILDLIFE MONITORING")
    print("="*70)
    print("\nTo run the test, use:")
    print("  locust -f locustfile.py --host=http://localhost:8000")
    print("\nThen open http://localhost:8089 in your browser")
    print("="*70 + "\n")
