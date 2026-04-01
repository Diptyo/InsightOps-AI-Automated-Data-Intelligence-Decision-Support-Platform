import os
import uuid
import shutil
import json 
import time
from flask import Flask, render_template, request, send_file, jsonify, Response, send_from_directory
from werkzeug.utils import secure_filename
from App import AIProjectEngine 

app = Flask(__name__)
app.secret_key = "executive_strategy_secret_key"

# --- DIRECTORY SETUP ---
UPLOAD_BASE = 'web_uploads'
REPORT_BASE = 'reports'
VISUALS_BASE = 'visuals'
for folder in [UPLOAD_BASE, REPORT_BASE, VISUALS_BASE]:
    os.makedirs(folder, exist_ok=True)

engine = AIProjectEngine()

# Global dictionary to store progress for different sessions
# Key: session_id, Value: current_status_string
progress_tracker = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/visuals/<filename>')
def serve_visual(filename):
    return send_from_directory(VISUALS_BASE, filename)

# --- SSE PROGRESS ROUTE ---
@app.route('/progress/<session_id>')
def progress(session_id):
    def generate():
        while True:
            # Check the global tracker for updates
            status = progress_tracker.get(session_id, "Initializing...")
            
            # Send the status to the frontend in SSE format
            yield f"data: {json.dumps({'status': status})}\n\n"
            
            # Stop the stream if task is finished or failed
            if status == "Complete" or "Error" in status:
                # Give the frontend a moment to receive the "Complete" status
                time.sleep(1)
                # Optional: clean up the tracker to save memory
                if session_id in progress_tracker:
                    del progress_tracker[session_id]
                break
            
            time.sleep(0.5) # Poll the internal dictionary every half second
            
    return Response(generate(), mimetype='text/event-stream')

# --- MAIN ANALYSIS ROUTE ---
@app.route('/analyze', methods=['POST'])
def analyze():
    # 1. Setup Session Identity
    # Frontend should ideally send a unique ID, otherwise we generate one
    session_id = request.form.get('session_id') or str(uuid.uuid4())[:8]
    session_dir = os.path.join(UPLOAD_BASE, session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    progress_tracker[session_id] = "Uploading Files..."

    # 2. Save Uploaded Files
    files = request.files.getlist('files')
    saved_paths = []
    for file in files:
        if file.filename != '':
            path = os.path.join(session_dir, secure_filename(file.filename))
            file.save(path)
            saved_paths.append(path)

    user_target = request.form.get('target_variable', "").strip() or None

    try:
        # Define the internal callback to update the global tracker
        def update_session_progress(status_message):
            print(f"[{session_id}] Status: {status_message}") # Log to terminal
            progress_tracker[session_id] = status_message

        # 3. Trigger the AI Engine
        # This will run sequentially but the /progress route can read the tracker
        report_data = engine.run_engine_web(
            file_paths=saved_paths, 
            manual_target=user_target, 
            session_id=session_id,
            progress_callback=update_session_progress
        )
        
        # 4. Finalize
        progress_tracker[session_id] = "Complete"
        
        # Clean up the raw CSV files after processing to save disk space
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)

        return jsonify({"success": True, "report": report_data, "session_id": session_id})
    
    except Exception as e:
        print(f"🔴 CRITICAL ERROR in /analyze: {str(e)}")
        progress_tracker[session_id] = f"Error: {str(e)}"
        return jsonify({"success": True, "report": report_data, "session_id": session_id})

@app.route('/download/<filename>')
def download_file(filename):
    path = os.path.join(REPORT_BASE, filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return "File not found", 404

if __name__ == '__main__':
    # threaded=True is vital for the progress bar to work while analysis is running
    app.run(debug=True, threaded=True, port=5000)