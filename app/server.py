from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

@app.route('/download_model', methods=['POST'])
def download_model():
  try:
    subprocess.run(['python', 'download_model.py'], check=True)
    return jsonify({"message": "Model downloaded successfully."}), 200
  except subprocess.CalledProcessError as e:
    return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)
