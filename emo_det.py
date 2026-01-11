from flask import Flask, jsonify, render_template
import subprocess
from recommendation import recommend_meal

# Flask automatically loads HTML from /templates/
app = Flask(__name__)


# ---------------------------
#           HOME PAGE
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------
#     START CAMERA ROUTE
# ---------------------------
@app.route("/start_camera")
def start_camera():
    try:
        print("[INFO] Launching emotion_webcame.py...")

        # Run your webcam detection script
        subprocess.run(["python", "emotion_webcame.py"], check=True)

        print("[INFO] Emotion detection finished. Preparing recommendation...")

        # Get emotion + recommended meals
        emotion, meals = recommend_meal()

        # Send result to frontend
        return jsonify({
            "emotion": emotion,
            "meals": meals
        })

    except subprocess.CalledProcessError as e:
        print("[ERROR] Webcam script failed:", e)
        return jsonify({"error": "Webcam script failed"}), 500

    except Exception as e:
        print("[ERROR]", e)
        return jsonify({"error": str(e)}), 500


# ---------------------------
#       MAIN ENTRY
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
