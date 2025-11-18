print("--- SCRIPT TOP LEVEL: main.py is starting ---")

import os
import sys
import firebase_admin
from firebase_admin import firestore, storage
import fitz  # PyMuPDF
import uvicorn
import json
from fastapi.staticfiles import StaticFiles
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import Counter
from types import SimpleNamespace 
import uuid

# ADK and Server Imports
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools import FunctionTool 
import google.generativeai as genai


# --- Global Demo Data Structures (For Local Demonstration) ---
# NOTE: This data now simulates live ICU patient and equipment data.

AMBULANCE_EQUIPMENT_DATA = {
    # This simulates an active ICU ambulance with patient data
    "AmbulanceID_A101_ICU": {
        "status": "In Transit - Patient Critical",
        "last_check": "2025-11-18T08:05:00Z",
        "defibrillator_charge": "95%",
        "oxygen_level": "90%",
        "medication_kit": ["Epinephrine (1)", "Midazolam (3)", "Fentanyl (1)"],
        "tire_pressure_alert": False,
        "patient_vitals": {
            "heart_rate_bpm": 115, # Tachycardic
            "blood_pressure_mmHg": "85/50", # Hypotensive
            "spo2_percent": 90 # Hypoxic
        },
        "icu_equipment": {
            "ventilator_mode": "AC/VC",
            "ventilator_pressure_cmH2O": 15,
            "infusion_pump_status": "Active - Norepinephrine @ 0.1 mcg/kg/min",
            "blood_gas_results_recent": "pH 7.28, pCO2 55, Base Deficit -5"
        }
    },
    # Standard ambulance (simplified for fleet check)
    "AmbulanceID_A102": {
        "status": "Maintenance Needed",
        "last_check": "2025-11-16T14:30:00Z",
        "defibrillator_charge": "15%",
        "oxygen_level": "95%",
        "medication_kit": ["Epinephrine (0)", "Aspirin (2)", "Bandages (Full)"],
        "tire_pressure_alert": True
    },
    "AmbulanceID_B203": {
        "status": "In Service",
        "last_check": "2025-11-17T08:00:00Z",
        "defibrillator_charge": "99%",
        "oxygen_level": "80%",
        "medication_kit": ["Epinephrine (2)", "Aspirin (10)", "Bandages (Full)"],
        "tire_pressure_alert": False
    },
}

# Assume this data comes from a Firestore collection of past calls
AMBULANCE_LOCATION_HISTORY = [
    {"timestamp": datetime.now() - timedelta(hours=1), "location": "Koramangala", "call_time_minutes": 5, "response_time_minutes": 10},
    {"timestamp": datetime.now() - timedelta(hours=3), "location": "Whitefield", "call_time_minutes": 15, "response_time_minutes": 25},
    {"timestamp": datetime.now() - timedelta(hours=5), "location": "Koramangala", "call_time_minutes": 8, "response_time_minutes": 12},
    {"timestamp": datetime.now() - timedelta(days=1), "location": "Jayanagar", "call_time_minutes": 7, "response_time_minutes": 11},
    {"timestamp": datetime.now() - timedelta(days=2), "location": "Whitefield", "call_time_minutes": 12, "response_time_minutes": 20},
    {"timestamp": datetime.now() - timedelta(days=3), "location": "Koramangala", "call_time_minutes": 6, "response_time_minutes": 9},
]


# --- Gemini and Firebase Setup (Unchanged) ---

print("--- Configuring GenAI and setting ENV VAR ---")
GEMINI_API_KEY = "AIzaSyBR1D3ztmCcKs7ivRIml-dV-G6UW1XdZvk"
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
print(f"--- Set OS Environ GEMINI_API_KEY to: {GEMINI_API_KEY[:4]}... ---")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    genai_model = genai.GenerativeModel(
        'gemini-2.0-flash', # Model used for internal summarization
        safety_settings=safety_settings
    )
    print("--- GenAI configured successfully (gemini-2.0-flash) ---")
except Exception as e:
    print(f"--- FATAL ERROR: genai.configure() or GenerativeModel() failed: {e} ---")
    sys.exit(1)


print("--- Initializing Firebase Admin ---")
try:
    if not firebase_admin._apps:
        # Replace with your actual Firebase project ID/bucket name
        STORAGE_BUCKET = 'medisync-1695206869761.appspot.com'
        firebase_admin.initialize_app(options={
            'storageBucket': STORAGE_BUCKET
        })
    
    db = firestore.client()
    bucket = storage.bucket() 
    print(f"Firebase Admin SDK initialized. Bucket: {bucket.name if bucket else 'None'}")
except Exception as e:
    print(f"Error initializing Firebase Admin: {e}")
    db = None
    bucket = None

APP_ID = 'medisync-1695206869761'


# --- Helper Function Definitions ---

def read_pdf_from_bytes(pdf_bytes: bytes) -> str:
    """Extracts text from a PDF provided as bytes."""
    print("--- read_pdf_from_bytes() called ---")
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        print(f"--- PDF read successfully, {len(text)} chars ---")
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def summarize_text(text: str, prompt: str) -> str:
    """Calls Gemini to summarize a chunk of text."""
    print(f"--- summarize_text() called with prompt: {prompt[:20]}... ---")
    try:
        response = genai_model.generate_content(
            f"{prompt}\n\n---\n\n{text}",
            generation_config=genai.types.GenerationConfig(
                temperature=0.2, 
            )
        )
        print("--- summarize_text() got response from Gemini ---")
        return response.text
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        return f"Error summarizing: {e}"

def create_pdf_from_text(text: str) -> bytes:
    """
    Creates a PDF file in memory from a string of text, titles it, and returns the bytes.
    Uses standard, guaranteed-available PostScript fonts (Times-Roman) to avoid deployment errors.
    """
    print("--- create_pdf_from_text() called ---")
    doc = fitz.open()
    page = doc.new_page()
    rect = page.rect
    
    # Define layout and fonts
    margin = 50
    font_size = 10
    title = "Consolidated Patient Medical Summary Report"
    
    # Calculate usable area for text
    usable_rect = fitz.Rect(margin, margin + 40, rect.width - margin, rect.height - margin)
    
    # 1. Insert Title - Using 'Times-Bold' (guaranteed PostScript font)
    page.insert_textbox(
        fitz.Rect(margin, margin, rect.width - margin, margin + 30),
        title, 
        fontname="Times-Bold", 
        fontsize=14, 
        align=fitz.TEXT_ALIGN_CENTER
    )
    
    # 2. Insert Body Text - Using 'Times-Roman' (guaranteed PostScript font)
    page.insert_textbox(
        usable_rect, 
        text, 
        fontname="Times-Roman", 
        fontsize=font_size, 
        align=fitz.TEXT_ALIGN_LEFT
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    print(f"--- PDF created in memory, size: {len(pdf_bytes)} bytes ---")
    return pdf_bytes


# --- 1. ADK Tool: Summarize Medical Reports (MODIFIED for PDF generation) ---

def summarize_all_user_reports(user_email: str) -> str:
    """
    Analyzes all PDF reports for a given user_email from Cloud Storage,
    returns a consolidated summary, AND saves the summary as a PDF to storage.
    """
    print(f"Tool called: summarize_all_user_reports for: {user_email}")
    if not bucket:
        return "Error: Firebase Storage is not initialized."

    try:
        folder_path = f"users/{user_email}/"
        blobs = list(bucket.list_blobs(prefix=folder_path))
        pdf_blobs = [b for b in blobs if b.name.lower().endswith(".pdf")]

        if not pdf_blobs:
            # If no actual reports are found, return a clear message, which will
            # be used as the body of the generated PDF.
            final_summary = (
                f"No PDF reports found for user '{user_email}' in Cloud Storage at path: {folder_path}. "
                f"Please ensure reports are uploaded to this location to generate a medical summary."
            )
            print("--- No PDF reports found, generating error summary. ---")
            
        else:

             # ENHANCED PROMPT: Define clear headings and a professional persona for data extraction.
            individual_summaries = []
            map_prompt = (
                "As an AI data extractor, summarize the following patient medical report. "
                "Output MUST be in markdown format with these three headings only: "
                "### Key Findings, ### Diagnoses, and ### Treatment Recommendations. "
                "Use bullet points under each heading."
            )
            print(f"--- Found {len(pdf_blobs)} PDF reports. Starting Map step (Gemini calls). ---")
            
            for blob in pdf_blobs:
                pdf_bytes = blob.download_as_bytes()
                report_text = read_pdf_from_bytes(pdf_bytes)
                if report_text:
                    summary = summarize_text(report_text, map_prompt)
                    individual_summaries.append(f"Summary of {blob.name}:\n{summary}")

            if not individual_summaries:
                return "Error: Could not read or summarize any of the PDF files."

            # --- REDUCE step (Consolidate summaries using Gemini) ---
            combined_summaries = "\n\n---\n\n".join(individual_summaries)
            reduce_prompt = (
                "You are a **Consulting Physician**. Review the set of individual patient report summaries provided below. "
                "Generate a **CONSOLIDATED, high-level clinical overview** (2-3 paragraphs) suitable for a chart review. "
                "The summary must synthesize all data, focusing on: "
                "1. **Current Status/Primary Problems:** The overarching medical condition and trajectory. "
                "2. **Long-Term Management:** A coherent summary of the unified treatment plan and next steps."
            )
            print("--- Starting Reduce step (Gemini call for final consolidation). ---")
            final_summary = summarize_text(combined_summaries, reduce_prompt)
        
        # --- PDF Generation and Upload (runs regardless of success to document the result) ---
        pdf_bytes = create_pdf_from_text(final_summary)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"consolidated_summary_{timestamp}.pdf"
        storage_path = f"users/{user_email}/summary_reports/{file_name}"
        
        blob = bucket.blob(storage_path)
        blob.upload_from_string(pdf_bytes, content_type='application/pdf')
        
        # Get the public URL for the file 
        public_url = f"https://storage.googleapis.com/{bucket.name}/{storage_path}"
        print(f"--- PDF uploaded successfully to: {public_url} ---")

        return f"Consolidated Summary for {user_email}: {final_summary}\n\n[PDF Report Saved: {public_url}]"

    except Exception as e:
        print(f"Error in summarize_all_user_reports: {e}")
        return f"Error: An exception occurred: {e}"

# --- 2. ADK Tool: Summarize Equipment Data (MODIFIED for ICU/Handover report) ---

def summarize_equipment_data(ambulance_id: Optional[str] = None) -> str:
    """
    If an ID for an ICU ambulance is provided, generates an urgent hospital handover report.
    Otherwise, summarizes the fleet equipment status.
    """
    print(f"Tool called: summarize_equipment_data for ID: {ambulance_id or 'ALL'}")
    
    data_to_summarize = {}
    if ambulance_id and ambulance_id in AMBULANCE_EQUIPMENT_DATA:
        data_to_summarize = {ambulance_id: AMBULANCE_EQUIPMENT_DATA[ambulance_id]}
    elif not ambulance_id:
        data_to_summarize = AMBULANCE_EQUIPMENT_DATA
    else:
        return f"Error: Ambulance ID '{ambulance_id}' not found."

    json_data = json.dumps(data_to_summarize, indent=2)

    # --- Conditional Prompt Selection ---
    
    # Check if a specific, patient-carrying ICU ambulance was requested
    is_icu_report = (
        ambulance_id and 
        ambulance_id in data_to_summarize and 
        'patient_vitals' in data_to_summarize[ambulance_id]
    )

    if is_icu_report:
        # Prompt for Urgent Hospital Handover Report (your use case)
     prompt = (
            "You are an emergency medical services (EMS) Paramedic reporting to a receiving hospital. "
            "Review the following JSON data, which represents the real-time status of the ambulance equipment AND critical patient vitals/interventions. "
            "Generate a CONCISE, urgent, two-paragraph summary suitable for a hospital handover. "
            "Paragraph 1 MUST focus on the patient: Summarize the patient's most critical vitals (HR, BP, SpO2), current interventions (ventilator, drips), and clinical state (e.g., hypotensive, acidotic), ensuring to highlight **actionable or concerning clinical details**. "
            "Paragraph 2 MUST focus on the equipment: Summarize the readiness of the ambulance equipment (defib charge, oxygen, medication), highlighting any maintenance needs or low inventory."
        )
    else:
        # Default Prompt for Fleet Management Summary
        prompt = (
            "You are an ambulance fleet manager. Review the following JSON data representing real-time "
            "equipment status. Identify all **URGENT and critical issues** (e.g., low battery, low oxygen, missing "
            "medication, maintenance status) and summarize them in a **clear, markdown table** (Ambulance ID, Issue, Severity). "
            "If all ambulances are healthy, state that clearly."
        )
    
    
    return summarize_text(json_data, prompt)

# --- 3. ADK Tool: Suggest Location Preferences (Unchanged) ---

def suggest_location_preferences(history_duration_days: int = 7) -> str:
    """
    Analyzes historical ambulance call data to suggest optimal deployment locations
    based on call density and average response times within a given time period.
    Default analysis period is 7 days.
    """
    print(f"Tool called: suggest_location_preferences for last {history_duration_days} days.")
    
    cutoff_date = datetime.now() - timedelta(days=history_duration_days)
    
    # 1. Filter data based on duration
    recent_calls = [
        call for call in AMBULANCE_LOCATION_HISTORY 
        if call['timestamp'] > cutoff_date
    ]
    
    if not recent_calls:
        return f"Error: No call history found in the last {history_duration_days} days for analysis."

    # 2. Aggregate data for LLM analysis
    location_counts = Counter(call['location'] for call in recent_calls)
    location_stats = {}
    
    for call in recent_calls:
        loc = call['location']
        if loc not in location_stats:
            location_stats[loc] = {'total_response_time': 0, 'call_count': 0}
        
        location_stats[loc]['total_response_time'] += call['response_time_minutes']
        location_stats[loc]['call_count'] += 1
    
    # Calculate average response time
    analysis_data = []
    for loc, data in location_stats.items():
        avg_response = data['total_response_time'] / data['call_count']
        analysis_data.append({
            "location": loc,
            "call_volume": data['call_count'],
            "average_response_time_minutes": round(avg_response, 2)
        })

    json_data = json.dumps(analysis_data, indent=2)

    prompt = (
        "You are a **Logistics and Deployment Analyst**. Review the following JSON data which contains "
        "ambulance call volume and average response times by location. "
        "Based on this data, provide a **single, specific, and actionable recommendation** for ambulance deployment. "
        "The recommendation MUST clearly state which locations need more coverage (high volume AND high response time) "
        "and justify the conclusion with the numerical evidence from the data."
    )

    return summarize_text(json_data, prompt)


# Create Tool Instances
summarize_tool_instance = FunctionTool(summarize_all_user_reports)
equipment_tool_instance = FunctionTool(summarize_equipment_data)
location_tool_instance = FunctionTool(suggest_location_preferences)


# --- ADK Agent Definition ---
print("--- Defining ADK Agent ---")
AGENT_INSTRUCTION = """
You are MediSync Agent, a **Critical Operations Support AI** specialized in emergency medical logistics and patient data synthesis. Your primary goal is to provide immediate, actionable intelligence to medical personnel and fleet managers.

You have access to the following three critical data tools:
1. `sync_summarize_all_user_reports(user_email)`: **Patient Chart Synthesis.** Use this tool when asked to generate a comprehensive medical overview or patient chart summary. You MUST provide the patient's `user_email`. The output includes a PDF link.
2. `summarize_equipment_data(ambulance_id)`: **Fleet & ICU Status Report.** Use this when the user asks for the health status of an ambulance fleet, or a detailed, urgent handover report for a specific, active **ICU ambulance** (requires `ambulance_id`).
3. `suggest_location_preferences(history_duration_days)`: **Deployment Optimization.** Use this when asked for strategic advice on where to station ambulances to reduce response times, based on historical call volume (optional number of days for analysis).

**Execution Protocol:**
* **PRIORITY:** Always attempt to map the user's request to one of the available tools.
* **Tool Use:** Call the required tool with the necessary parameters extracted from the user's prompt.
* **Error Handling:** If a tool execution fails, immediately return the error message provided by the tool.
* **General Queries:** For questions unrelated to your tools, answer concisely and remind the user of your specialized function (patient records, equipment, or deployment).
"""


try:
    summarizer_agent = Agent(
        name="medisync_agent",
        # Model for the Agent's reasoning and tool selection
        model=LiteLlm("gemini/gemini-2.0-flash"), 
        instruction=AGENT_INSTRUCTION,
        # List of ALL available tools for the agent to select from
        tools=[
            summarize_tool_instance,
            equipment_tool_instance,
            location_tool_instance
        ], 
    )
    print("--- ADK Agent defined successfully ---")
except Exception as e:
    print(f"--- FATAL ERROR: Agent() or LiteLlm() failed: {e} ---")
    sys.exit(1)

# --- FastAPI Server Setup (Unchanged) ---
print("--- Setting up FastAPI Server ---")
app = FastAPI(title="MediSync Agent")
session_service = InMemorySessionService()

runner = Runner(
    agent=summarizer_agent, 
    app_name="medisync", 
    session_service=session_service
)


# --- NEW: Route to serve index.html from the root URL ---
@app.get("/", include_in_schema=False)
async def serve_index():
    """
    Serves the index.html file when the user accesses the root URL.
    This ensures the web app loads without requiring /index.html in the URL.
    """
    index_file = "index.html"
    if os.path.exists(index_file):
        # We use FileResponse to serve the HTML file directly
        return FileResponse(index_file, media_type="text/html")
    else:
        # Fallback to a simple message if index.html is missing
        return {"status": "ok", "service": "MediSync Agent Running (Index not found)"}



@app.post("/agent/run")
async def run_agent(request: Request):
    """
    Endpoint to trigger the agent. Expects a JSON payload with 'user_id' and 'prompt'.
    """
    try:
        data = await request.json()
        
        # The user_id is primarily for session/tracking, but we'll use a dynamic ID
        user_id = data.get("user_id", str(uuid.uuid4())) 
        user_prompt = data.get("prompt")
        
        if not user_prompt:
            raise HTTPException(status_code=400, detail="prompt is required")

        print(f"--- Received request for {user_id} with prompt: {user_prompt} ---")

        # Create a unique session for the request
        session_id = f"session_{user_id}_{uuid.uuid4()}"
        
        await session_service.create_session(
            app_name="medisync",
            user_id=user_id,
            session_id=session_id
        )

        # Create a mock Message Object using SimpleNamespace
        part_object = SimpleNamespace(text=user_prompt)
        message_object = SimpleNamespace(role="user", parts=[part_object])

        final_response_text = ""
        
        # Run the agent and capture all output events
        for event in runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=message_object 
        ):
            print(f"DEBUG EVENT: {event}")
            
            # The structure of the event text depends on the ADK version
            if hasattr(event, 'text') and event.text:
                 final_response_text += event.text
            elif hasattr(event, 'content') and event.content:
                 final_response_text += str(event.content)
            elif hasattr(event, 'payload') and isinstance(event.payload, dict):
                 if 'text' in event.payload:
                     final_response_text += event.payload['text']
                 elif 'content' in event.payload:
                     final_response_text += str(event.payload['content'])

        print(f"--- Agent execution finished. Captured text length: {len(final_response_text)} ---")
        
        if not final_response_text:
            final_response_text = "The agent ran but produced no direct text output. Check server logs."

        return {"response": final_response_text, "status": "success"}

    except Exception as e:
        print(f"Server Error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
 
#@app.get("/")
#async def health_check():
 #   return {"status": "ok", "service": "MediSync Agent Running"}

#print("--- ADK Server setup complete. ---")

#app.mount("/", StaticFiles(directory=".", html=True), name="static")
app.mount("/", StaticFiles(directory=".", html=True), name="static")

print("--- ADK Server setup complete, including static file server. ---")

if __name__ == '__main__':
    print(f"--- Running __main__ block ---")
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting ADK server locally on port {port}...")
    uvicorn.run(app, host='0.0.0.0', port=port)
