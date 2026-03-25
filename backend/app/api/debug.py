"""
MiroFish Debug API - Log streaming & control commands
"""

import json
import logging
import os
import time
from collections import deque
from datetime import datetime

import psutil
from flask import Response, jsonify, request, stream_with_context

from . import debug_bp
from ..utils.logger import get_logger

logger = get_logger('mirofish.debug')

# ---------------------------------------------------------------------------
# In-memory log buffer + custom handler
# ---------------------------------------------------------------------------

_log_buffer = deque(maxlen=1000)
_start_time = time.time()


class BufferedLogHandler(logging.Handler):
    """Captures log records into a shared deque for SSE streaming."""

    def emit(self, record):
        entry = {
            'timestamp': datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3],
            'level': record.levelname,
            'message': self.format(record),
        }
        _log_buffer.append(entry)


# Attach handler to root 'mirofish' logger so we capture everything
_handler = BufferedLogHandler()
_handler.setFormatter(logging.Formatter('%(message)s'))
logging.getLogger('mirofish').addHandler(_handler)


# ---------------------------------------------------------------------------
# SSE log streaming
# ---------------------------------------------------------------------------

@debug_bp.route('/logs')
def stream_logs():
    """GET /api/debug/logs — Server-Sent Events log stream."""

    def _generate():
        idx = len(_log_buffer)
        # Send existing buffer first
        for entry in list(_log_buffer):
            yield f"data: {json.dumps(entry, ensure_ascii=False)}\n\n"
        # Then poll for new entries
        while True:
            current_len = len(_log_buffer)
            if current_len > idx:
                for entry in list(_log_buffer)[idx:current_len]:
                    yield f"data: {json.dumps(entry, ensure_ascii=False)}\n\n"
                idx = current_len
            else:
                yield ": keepalive\n\n"
            time.sleep(0.5)

    return Response(
        stream_with_context(_generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Access-Control-Allow-Origin': '*',
        },
    )


# ---------------------------------------------------------------------------
# Quick status
# ---------------------------------------------------------------------------

def _build_status():
    mem = psutil.virtual_memory()
    sim_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info.get('cmdline') or [])
            if 'run_parallel_simulation' in cmdline or 'mirofish' in cmdline.lower():
                sim_processes.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return {
        'uptime_seconds': round(time.time() - _start_time),
        'memory': {
            'total_gb': round(mem.total / (1024 ** 3), 1),
            'used_gb': round(mem.used / (1024 ** 3), 1),
            'percent': mem.percent,
        },
        'cpu_percent': psutil.cpu_percent(interval=0.1),
        'simulation_processes': sim_processes,
        'active_simulations': len(sim_processes),
        'log_buffer_size': len(_log_buffer),
    }


@debug_bp.route('/status')
def status():
    """GET /api/debug/status — System status snapshot."""
    return jsonify(_build_status())


# ---------------------------------------------------------------------------
# Command execution
# ---------------------------------------------------------------------------

_COMMANDS = {
    '/kill': 'Kill all simulation processes',
    '/restart': 'Kill processes and signal restart',
    '/status': 'Show system status',
    '/clear': 'Clear log buffer',
    '/config': 'Show current config (API keys masked)',
    '/help': 'List available commands',
}


def _cmd_kill():
    """Terminate all MiroFish simulation sub-processes."""
    killed = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info.get('cmdline') or [])
            if 'run_parallel_simulation' in cmdline:
                proc.terminate()
                killed.append(proc.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Also try SimulationRunner cleanup
    try:
        from ..services.simulation_runner import SimulationRunner
        SimulationRunner.register_cleanup()
    except Exception:
        pass

    logger.info(f"Killed {len(killed)} simulation process(es): {killed}")
    return {'killed': killed, 'count': len(killed)}


def _cmd_restart():
    result = _cmd_kill()
    result['restart'] = True
    result['message'] = 'Processes killed. Restart the server manually or via npm run dev.'
    return result


def _cmd_status():
    return _build_status()


def _cmd_clear():
    _log_buffer.clear()
    logger.info("Log buffer cleared")
    return {'message': 'Log buffer cleared'}


def _cmd_config():
    """Return .env values with API keys masked."""
    env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    config = {}
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    # Mask sensitive values
                    if any(s in key.upper() for s in ['KEY', 'SECRET', 'TOKEN', 'PASSWORD']):
                        value = '****' + value[-4:] if len(value) > 4 else '****'
                    config[key] = value
    except FileNotFoundError:
        return {'error': '.env file not found'}
    return config


def _cmd_help():
    return {'commands': _COMMANDS}


_COMMAND_MAP = {
    '/kill': _cmd_kill,
    '/restart': _cmd_restart,
    '/status': _cmd_status,
    '/clear': _cmd_clear,
    '/config': _cmd_config,
    '/help': _cmd_help,
}


@debug_bp.route('/command', methods=['POST'])
def execute_command():
    """POST /api/debug/command — Execute a control command."""
    data = request.get_json(silent=True) or {}
    cmd = data.get('command', '').strip().lower()

    if not cmd:
        return jsonify({'error': 'No command provided'}), 400

    # Allow commands with or without leading /
    if not cmd.startswith('/'):
        cmd = '/' + cmd

    handler = _COMMAND_MAP.get(cmd)
    if not handler:
        return jsonify({
            'error': f'Unknown command: {cmd}',
            'available': list(_COMMANDS.keys()),
        }), 400

    try:
        result = handler()
        return jsonify({'command': cmd, 'result': result})
    except Exception as e:
        logger.error(f"Command {cmd} failed: {e}")
        return jsonify({'command': cmd, 'error': str(e)}), 500
