from flask import Flask, request, jsonify
import logging
from datetime import datetime

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/chat-assistant', methods=['POST'])
def chat_assistant():
    """
    Endpoint to handle chat assistant commands.
    Expected JSON format:
    {
        "wakeword": "hey friday",
        "command": "open the door"
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        if not data or 'wakeword' not in data or 'command' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing required fields: wakeword and command are required'
            }), 400
        
        wakeword = data['wakeword'].strip().lower()
        command = data['command'].strip().lower()
        
        # Log the request
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'wakeword': wakeword,
            'command': command,
            'ip': request.remote_addr
        }
        logger.info(f"Chat assistant request: {log_entry}")
        
        # Process the command (add your custom logic here)
        response = process_command(command)
        
        return jsonify({
            'status': 'success',
            'message': 'Command processed successfully',
            'command': command,
            'response': response,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'An error occurred: {str(e)}'
        }), 500

def process_command(command):
    """Process the command and return a response."""
    # Add your command processing logic here
    command = command.lower()
    
    if 'open' in command and 'door' in command:
        return "The door is now open."
    elif 'close' in command and 'door' in command:
        return "The door is now closed."
    elif 'time' in command:
        return f"The current time is {datetime.now().strftime('%H:%M:%S')}."
    elif 'date' in command:
        return f"Today's date is {datetime.now().strftime('%Y-%m-%d')}."
    else:
        return f"I received your command: '{command}'. This is a default response."

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
