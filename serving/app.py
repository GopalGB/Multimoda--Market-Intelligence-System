#!/usr/bin/env python
import os
import logging
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required, create_access_token
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import routes
from api.routes import api_bp
from api.auth import auth_bp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the Flask application."""
    # Initialize Flask app
    app = Flask(__name__)
    
    # Load configuration
    app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'dev-key-change-in-production')
    app.config['ENVIRONMENT'] = os.getenv('ENVIRONMENT', 'development')
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = 3600  # 1 hour
    
    # Configure CORS
    allowed_origins = os.getenv('ALLOWED_ORIGINS', '*').split(',')
    CORS(app, resources={r"/api/*": {"origins": allowed_origins}})
    
    # Initialize JWT
    jwt = JWTManager(app)
    
    # Initialize rate limiter
    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=[os.getenv('RATE_LIMIT', '100/hour')],
        storage_uri="memory://"
    )
    
    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api/v1')
    app.register_blueprint(auth_bp, url_prefix='/auth')
    
    # Root route for API health check
    @app.route('/')
    def index():
        return jsonify({
            'status': 'ok',
            'service': 'Audience Intelligence API',
            'version': '1.0.0',
            'environment': app.config['ENVIRONMENT'],
            'timestamp': datetime.now().isoformat()
        })
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': 'Not Found',
            'message': 'The requested resource does not exist'
        }), 404
    
    @app.errorhandler(500)
    def server_error(error):
        logger.error(f"Server error: {error}")
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred'
        }), 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 5000))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    # Start the server
    logger.info(f"Starting Audience Intelligence API on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug) 