MediSync Cognitive AI Engine (Cloud Run Service)

The MediSync Cognitive AI Engine is the intelligence core of the Smart Ambulance Service. It is a serverless, highly-scalable backend service deployed on Google Cloud Run that uses the Google Agent Development Kit (ADK) and the Gemini API to process real-time and historical medical data, enabling critical, rapid decision-making for ambulance dispatch, patient care, and hospital triage.

🌟 Key Capabilities & AI Modules

This service orchestrates the complex logic required for a truly intelligent emergency response system:

AI Module

Function

Technology

Health Data Mentor

Uses Gemini 2.0 (LLM/NLP) to analyze unstructured patient data (PDFs, reports from Firebase Storage) and generate a concise, encrypted Health Snapshot Summary.

ADK, Gemini API, PyMuPDF

In-Transit Data Synthesis

Processes high-frequency vitals (telemetry) alongside the Health Snapshot to synthesize real-time Critical Condition Summaries and flag escalating risks.

ADK, Firebase Firestore, Time-Series Logic

Smart Allocation & Route

Consumes live location data (Google Maps SDK integration) and ambulance status (Firestore) to recommend the optimally equipped and closest vehicle.

FastAPI Logic, Google Maps API

Intelligent Hospital Matching

Matches the patient's immediate medical needs (from the Critical Summary) with real-time hospital resource availability (beds, specialized equipment) to suggest the best facility.

FastAPI Logic, Firestore

🛠️ Technology Stack

Component

Technology

Role

AI Orchestration

Google ADK (Agent Development Kit)

Manages the multi-step, tool-using AI logic (e.g., Map-Reduce for summarizing reports).

Core LLM

Google AI Gemini 2.0

Powers the core analytical and summarization tasks (e.g., Health Data Mentor).

Backend Framework

Python, FastAPI, Uvicorn

Provides the high-performance, asynchronous web server (main.py).

Deployment

Google Cloud Run

Serverless platform for scalable, containerized deployment.

Database/Storage

Firebase (Firestore, Storage)

Persistent storage for patient records (PHI) and real-time data ingestion.

Mobile Integration

Android SDK, Google Maps SDK

Frontend interfaces for patients and ambulance drivers that communicate with this service.

🚀 Setup & Deployment

1. Prerequisites

Python 3.10+

Docker (for containerization)

A Google Cloud Project with Billing Enabled

Authenticated Google Cloud SDK (gcloud auth login and gcloud config set project [YOUR-PROJECT-ID])

A GEMINI_API_KEY environment variable (or hardcoded in main.py for local testing).

A Firebase service account key (or relying on Cloud Run's default service account).

2. Local Development

Clone the repository:

git clone [REPO-URL]
cd medisync-ai-core


Install dependencies:
The core dependencies are listed in requirements.txt:

uv pip install -r requirements.txt


Run the FastAPI server:
The application runs on port 8080 by default:

python main.py


The service will be available at http://localhost:8080.

3. Deploying to Cloud Run

We use the provided Dockerfile to containerize the application for serverless deployment.

Build the Docker Image:

export PROJECT_ID=$(gcloud config get-value project)
export SERVICE_NAME=medisync-ai-agent

docker build -t gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest .
docker push gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest


Deploy to Cloud Run:
Deploy the image as a new service. Ensure you select a region and allow unauthenticated invocations (or configure IAM appropriately).

gcloud run deploy ${SERVICE_NAME} \
  --image gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest \
  --platform managed \
  --region [YOUR-REGION] \
  --allow-unauthenticated \
  --memory 2Gi # Recommended for PyMuPDF/AI tasks


Configure Environment Variables (Crucial):
In the Cloud Run console for your deployed service, set the following environment variables:

GEMINI_API_KEY: Your key for the Gemini API.

APP_ID: Your Firebase/GCP Project ID.

📝 API Endpoint Usage

The primary interaction with the AI Core is via the /invoke endpoint, which is designed to accept a specific request (query) and a context ID (session/user).

POST /invoke

This endpoint triggers the ADK agent's execution path (e.g., retrieving patient files, running the Map-Reduce summarization, or finding a suitable hospital).

Method: POST

Content-Type: application/json

Example Request (Simulating a Doctor Request):

{
  "session_id": "patient-xyz-456",
  "query": "Generate a concise critical summary for the current patient's vitals, integrating the historical health snapshot. Also, recommend the best facility within 10km for a cardiovascular trauma."
}


Response Format:

{
  "response": "Based on live vitals (HR: 115, BP: 90/60) and history (known cardiac issues), the patient is in critical but stable condition. **Optimal Hospital Recommendation:** City General (8.5 km, 20 min drive). They have 2 ICU beds and a Cardio-Trauma specialist available. Summary PDF generated and sent to triage console.",
  "status": "success"
}
