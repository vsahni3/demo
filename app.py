from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin  # Import CORS handling
from process_video import VideoSummary, VideoSummarySegment, reply, upload_video


app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB




@app.route('/respond', methods=['GET'])
def respond():
    video = open('recording.mov', 'rb')
    video = upload_video(video)
    response = reply(video)
    urls = set()
    for snapshot in response.segments:
        for url in snapshot.visited_urls:
            urls.add(url)

    for url in urls:
        html = get_html(url)
        markdown = md(html)
        chunk_and_store_markdown(url, markdown)
    query = response.user_intention
    docs_with_score = get_k_most_relevant(query, 20)
    docs = [doc[0] for doc in docs_with_score]
    instructions = format_instructions(response, docs)
    return jsonify({"response": instructions}), 200



if __name__ == '__main__':
    app.run(debug=True, port=5002)

