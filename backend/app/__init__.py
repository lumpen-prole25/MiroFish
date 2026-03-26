"""
MiroFish Backend - Flask应用工厂
"""

import os
import warnings

# 抑制 multiprocessing resource_tracker 's warning(来自round 三方库such as transformers)
# 需要in hasits他导入之前setup
warnings.filterwarnings("ignore", message=".*resource_tracker.*")

from flask import Flask, request
from flask_cors import CORS

from .config import Config
from .utils.logger import setup_logger, get_logger


def create_app(config_class=Config):
    """Flask应用工厂function"""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # SetJSON编码:ensure文直接显示(而不is  \uXXXX format)
    # Flask >= 2.3 use app.json.ensure_ascii, 旧versionuse JSON_AS_ASCII configuration
    if hasattr(app, 'json') and hasattr(app.json, 'ensure_ascii'):
        app.json.ensure_ascii = False
    
    # Setlog
    logger = setup_logger('mirofish')
    
    # onlyin  reloader 子process打印start信息(避免 debug mode下打印two times)
    is_reloader_process = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
    debug_mode = app.config.get('DEBUG', False)
    should_log_startup = not debug_mode or is_reloader_process
    
    if should_log_startup:
        logger.info("=" * 50)
        logger.info("MiroFish Backend Starting...")
        logger.info("=" * 50)
    
    # enableCORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Registersimulationprocesscleanupfunction(ensureserverclosetime终止hassimulationprocess)
    from .services.simulation_runner import SimulationRunner
    SimulationRunner.register_cleanup()
    if should_log_startup:
        logger.info("Registered simulation process cleanup function")
    
    # requestlog间件
    @app.before_request
    def log_request():
        logger = get_logger('mirofish.request')
        logger.debug(f"request: {request.method} {request.path}")
        if request.content_type and 'json' in request.content_type:
            logger.debug(f"request体: {request.get_json(silent=True)}")
    
    @app.after_request
    def log_response(response):
        logger = get_logger('mirofish.request')
        logger.debug(f"response: {response.status_code}")
        return response
    
    # Register蓝图
    from .api import graph_bp, simulation_bp, report_bp, debug_bp
    app.register_blueprint(graph_bp, url_prefix='/api/graph')
    app.register_blueprint(simulation_bp, url_prefix='/api/simulation')
    app.register_blueprint(report_bp, url_prefix='/api/report')
    app.register_blueprint(debug_bp, url_prefix='/api/debug')
    
    # Health check
    @app.route('/health')
    def health():
        return {'status': 'ok', 'service': 'MiroFish Backend'}
    
    if should_log_startup:
        logger.info("MiroFish Backend startcomplete")
    
    return app

