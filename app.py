from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin  # Import CORS handling
from process_video import VideoSummary, VideoSummarySegment, reply, upload_video
import os

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

@app.route('/respond', methods=['GET'])
def respond():
    video_path = 'demo/recording.mov'
    if not os.path.exists(video_path):
        return jsonify({"error": "Video file not found"}), 404

    with open(video_path, 'rb') as video:
        video_file = upload_video(video)
    response = reply(video_file)


    return jsonify(response), 200

if __name__ == '__main__':
    app.run(debug=False, port=5002)
