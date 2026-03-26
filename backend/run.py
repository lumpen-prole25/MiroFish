"""
MiroFish Backend start入口
"""

import os
import sys

# 解决 Windows 控制台文乱码question:in has导入之前setup UTF-8 编码
if sys.platform == 'win32':
    # setup环境变量ensure Python use UTF-8
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    # 重新config标准output流 UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# add items目root directoryto 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.config import Config


def main():
    """Main function"""
    # validateconfig
    errors = Config.validate()
    if errors:
        print("Configuration errors:")
        for err in errors:
            print(f"  - {err}")
        print("\n请check .env 文件's config")
        sys.exit(1)
    
    # create应用
    app = create_app()
    
    # Getrunningconfig
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5001))
    debug = Config.DEBUG
    
    # Starting service
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    main()

