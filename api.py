from flask import Flask, request, jsonify
import logging
import requests
from threading import Thread
from chatbot_qdrant import connect_qdrant, create_collection_if_not_exists, chatbot_rag
from embeddings import load_model
from config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, VECTOR_SIZE, SLACK_BOT_TOKEN

# Slack bot token

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__)

    # -------------------------
    # Load Qdrant + embedding 1 lần
    # -------------------------
    logger.info("Connecting to Qdrant...")
    client = connect_qdrant(QDRANT_URL, QDRANT_API_KEY)
    create_collection_if_not_exists(client, COLLECTION_NAME, VECTOR_SIZE)
    logger.info("Loading embedding model...")
    embedding_model = load_model()
    logger.info("Qdrant + Embedding ready!")

    # -------------------------
    # Slack DM function
    # -------------------------
    def send_slack_dm(user_id: str, message_text: str):
        try:
            resp = requests.post(
                "https://slack.com/api/chat.postMessage",
                headers={
                    "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
                    "Content-type": "application/json"
                },
                json={"channel": user_id, "text": message_text}
            ).json()
            logger.info(f"Slack API response: {resp}")
            return resp.get("ok", False)
        except Exception as e:
            logger.error(f"Error sending Slack DM: {e}")
            return False

    # -------------------------
    # Background handler
    # -------------------------
    def handle_event(user_id, text):
        logger.info(f"Processing message from {user_id}: {text}")
        try:
            answer, source = chatbot_rag(client, text, embedding_model)
            response_text = f"🤖 ChatBot trả lời:\n{answer}\n📄 Nguồn: {source}"
        except Exception as e:
            logger.error(f"chatbot_rag error: {e}")
            response_text = "⚠️ Bot gặp lỗi, thử lại sau."
        send_slack_dm(user_id, response_text)

    # -------------------------
    # Slack events endpoint
    # -------------------------
    @app.route("/slack/events", methods=["POST"], strict_slashes=False)
    def slack_events():
        data = request.json
        if not data:
            return "Bad Request", 400

        # URL verification
        if data.get("type") == "url_verification":
            return jsonify({"challenge": data["challenge"]})

        # Event callback
        if data.get("type") == "event_callback":
            event = data.get("event", {})

            # Only handle user messages (ignore bot messages, edits, etc.)
            if event.get("type") == "message" and "subtype" not in event and event.get("channel_type") == "im":
                user_id = event.get("user")
                text = event.get("text")

                # Dispatch thread for **each message separately**
                Thread(target=handle_event, args=(user_id, text), daemon=True).start()

        # Return 200 OK immediately to Slack
        return jsonify({"status": "ok"}), 200

    # -------------------------
    # Test endpoint
    # -------------------------
    @app.route("/test", methods=["GET"])
    def test():
        return "Chatbot Slack API is running!"

    return app

if __name__ == "__main__":
    app = create_app()
    # disable reloader to avoid double-loading
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
