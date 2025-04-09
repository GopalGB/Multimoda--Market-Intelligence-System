"""
Authentication module for the Audience Intelligence API.
Handles user registration, login, and API key management.
"""
import os
import logging
import secrets
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify
from flask_jwt_extended import (
    create_access_token, jwt_required, get_jwt_identity
)
from pydantic import BaseModel, EmailStr, validator, Field
from typing import Optional, Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
auth_bp = Blueprint('auth', __name__)

# Mock database for users and API keys (replace with real DB in production)
users_db = {}
api_keys = {}

# Validation models
class UserRegistration(BaseModel):
    email: EmailStr
    password: str
    name: str
    
    @validator('password')
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class ApiKeyRequest(BaseModel):
    description: str = Field(..., min_length=3, max_length=100)

# Routes
@auth_bp.route('/register', methods=['POST'])
def register():
    """Register a new user."""
    try:
        data = request.get_json()
        user = UserRegistration(**data)
        
        # Check if user already exists
        if user.email in users_db:
            return jsonify({
                'error': 'User already exists',
                'message': 'A user with this email already exists'
            }), 409
        
        # Store user (hash password in production)
        users_db[user.email] = {
            'name': user.name,
            'password': user.password,  # NEVER store plaintext in production
            'created_at': datetime.now().isoformat()
        }
        
        # Create access token
        access_token = create_access_token(identity=user.email)
        
        return jsonify({
            'message': 'User registered successfully',
            'access_token': access_token
        }), 201
        
    except ValueError as e:
        return jsonify({
            'error': 'Validation Error',
            'message': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({
            'error': 'Server Error',
            'message': 'An error occurred during registration'
        }), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    """Login a user."""
    try:
        data = request.get_json()
        login_data = UserLogin(**data)
        
        # Check if user exists
        if login_data.email not in users_db:
            return jsonify({
                'error': 'Authentication Failed',
                'message': 'Invalid email or password'
            }), 401
        
        # Verify password (use secure comparison in production)
        user = users_db[login_data.email]
        if user['password'] != login_data.password:
            return jsonify({
                'error': 'Authentication Failed',
                'message': 'Invalid email or password'
            }), 401
        
        # Create access token
        access_token = create_access_token(identity=login_data.email)
        
        return jsonify({
            'message': 'Login successful',
            'access_token': access_token,
            'user': {
                'email': login_data.email,
                'name': user['name']
            }
        }), 200
        
    except ValueError as e:
        return jsonify({
            'error': 'Validation Error',
            'message': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({
            'error': 'Server Error',
            'message': 'An error occurred during login'
        }), 500

@auth_bp.route('/api-keys', methods=['POST'])
@jwt_required()
def create_api_key():
    """Create a new API key for the authenticated user."""
    try:
        user_email = get_jwt_identity()
        data = request.get_json()
        key_request = ApiKeyRequest(**data)
        
        # Generate API key
        api_key = secrets.token_urlsafe(32)
        
        # Store API key
        if user_email not in api_keys:
            api_keys[user_email] = []
        
        api_keys[user_email].append({
            'key': api_key,
            'description': key_request.description,
            'created_at': datetime.now().isoformat()
        })
        
        return jsonify({
            'message': 'API key created successfully',
            'api_key': api_key
        }), 201
        
    except ValueError as e:
        return jsonify({
            'error': 'Validation Error',
            'message': str(e)
        }), 400
    except Exception as e:
        logger.error(f"API key creation error: {e}")
        return jsonify({
            'error': 'Server Error',
            'message': 'An error occurred while creating API key'
        }), 500

@auth_bp.route('/api-keys', methods=['GET'])
@jwt_required()
def list_api_keys():
    """List all API keys for the authenticated user."""
    try:
        user_email = get_jwt_identity()
        
        # Get user's API keys
        user_keys = api_keys.get(user_email, [])
        
        # Return redacted keys (don't expose full key)
        redacted_keys = []
        for key_data in user_keys:
            # Only show first 8 and last 4 characters
            key = key_data['key']
            redacted = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "..."
            
            redacted_keys.append({
                'key_preview': redacted,
                'description': key_data['description'],
                'created_at': key_data['created_at']
            })
        
        return jsonify({
            'api_keys': redacted_keys
        }), 200
        
    except Exception as e:
        logger.error(f"API key listing error: {e}")
        return jsonify({
            'error': 'Server Error',
            'message': 'An error occurred while listing API keys'
        }), 500

@auth_bp.route('/status', methods=['GET'])
def status():
    """Public endpoint to check authentication service status."""
    return jsonify({
        'status': 'operational',
        'service': 'authentication',
        'timestamp': datetime.now().isoformat()
    }), 200 